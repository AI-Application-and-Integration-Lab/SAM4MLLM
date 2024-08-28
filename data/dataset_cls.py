import random
from PIL import Image

import torch
from torch.utils.data import Dataset

from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

SYSTEM_PROMPT = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."

class RefDataset(Dataset):
    def __init__(self, data, tokenizer, img_processor, model_config, n_points=10,
                 img_dir='/home/ai2lab/work/datasets/', system_prompts=None):
        self.data = data
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.model_config = model_config
        self.n_points = n_points
        self.img_dir = img_dir
        
        self.system_prompts = [SYSTEM_PROMPT] if system_prompts is None else system_prompts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example_d = self.data[idx]
        image_path = example_d['image_path']
        image = Image.open(self.img_dir + image_path).resize((672, 672))

        conv = [{"role": "system", "content": random.choice(self.system_prompts)}]

        s_phrase = random.choice(example_d['phrases'])
        answer_counts = example_d['answer_counts']
        question_box = f'<image>\nPlease provide the bounding box coordinate of the region this sentence describes ({answer_counts}):\n"{s_phrase}".'
        if len(example_d['bboxes']) == 0:
            answer_box = 'No object found.'
        else:
            answer_box = ' '.join([f'[{x[0]},{x[1]},{x[2]},{x[3]}]' for x in example_d['bboxes']])

        conv.extend([
            {"role": "user", "content": question_box},
            {"role": "assistant", "content": answer_box}
        ])

        for bbox, p_n_ls in zip(example_d['bboxes'], example_d['points_and_labels']):
            n_sel_points = random.randint(10, 30)
            sampled_points_and_labels = random.sample(p_n_ls, n_sel_points)
            
            points_txt = ' '.join([f'[{x[0]},{x[1]}]' for x in sampled_points_and_labels])
            question_points = 'Check if the points listed below are located on the object with coordinates [{},{},{},{}]:\n{}'.format(
                bbox[0], bbox[1], bbox[2], bbox[3], points_txt)
            answer_points = ''.join(['Yes' if x[2] else 'No' for x in sampled_points_and_labels])
            
            conv.extend([
                {"role": "user", "content": question_points},
                {"role": "assistant", "content": answer_points}
            ])
            
        conv_text = self.tokenizer.apply_chat_template(conv, tokenize=False)
        input_ids = tokenizer_image_token(conv_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        
        pixel_values = process_images([image], self.img_processor, self.model_config)[0].half()
        image_sizes = [672, 672]

        sample = {
            'input_ids': input_ids,
            'images': pixel_values,
            'image_sizes': image_sizes
        }
        
        return sample


class GroupRefDataset(Dataset):
    def __init__(
            self,
            img_grouped_data,
            tokenizer,
            img_processor,
            model_config,
            n_points=10,
            img_dir='/home/ai2lab/work/datasets/',
            max_len=4096,
            img_size=(672,672),
            system_prompts=None
        ):
        self.img_grouped_data = img_grouped_data
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.model_config = model_config
        self.n_points = n_points
        self.img_dir = img_dir
        self.max_len = max_len
        self.img_size = img_size
        
        if system_prompts is None:
            self.system_prompts = [SYSTEM_PROMPT]
        
        self.ref_data = []
        self.reload(0)

    def __len__(self):
        return len(self.ref_data)
    
    def reload(self, seed):
        self.ref_data = []
        n_ref_per_img = 6
        
        print(f'Dataset reloaded with seed {seed}')
        random.seed(seed)
        for img_path, refs in self.img_grouped_data.items():
            random.shuffle(refs)
            for i in range(0, len(refs), n_ref_per_img):
                self.ref_data.append(refs[i:i+n_ref_per_img])

    def __getitem__(self, idx, tokenize=True):
        ref_samples = self.ref_data[idx]
        image_path = ref_samples[0]['image_path']
        image = Image.open(self.img_dir + image_path).resize(self.img_size)

        conv = [{"role": "system", "content": random.choice(self.system_prompts)}]
        
        for i_conv, ref_sample in enumerate(ref_samples):

            s_phrase = random.choice(ref_sample['phrases'])
            answer_counts = ref_sample['answer_counts']
            question_box = '<image>\n' if i_conv == 0 else ''
            question_box += f'Please provide the bounding box coordinate of the region this sentence describes ({answer_counts}):\n"{s_phrase}".'
            if len(ref_sample['bboxes']) == 0:
                answer_box = 'No object found.'
            else:
                answer_box = ' '.join([f'[{x[0]},{x[1]},{x[2]},{x[3]}]' for x in ref_sample['bboxes']])

            conv.extend([
                {"role": "user", "content": question_box},
                {"role": "assistant", "content": answer_box}
            ])

            for bbox, p_n_ls in zip(ref_sample['bboxes'], ref_sample['points_and_labels']):
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 1000 / 1000
                n_sel_points = random.randint(10, 30)
                sampled_points_and_labels = random.sample(p_n_ls, n_sel_points)
                
                points_txt = ' '.join([f'[{x[0]},{x[1]}]' for x in sampled_points_and_labels])
                question_points = 'Check if the points listed below are located on the object with coordinates [{},{},{},{}]:\n{}'.format(
                    bbox[0], bbox[1], bbox[2], bbox[3], points_txt)
                answer_points = ''.join(['Yes' if x[2] else 'No' for x in sampled_points_and_labels])
                
                conv.extend([
                    {"role": "user", "content": question_points},
                    {"role": "assistant", "content": answer_points}
                ])
            
        conv_text = self.tokenizer.apply_chat_template(conv, tokenize=False)
        
        if tokenize:
            input_ids = tokenizer_image_token(conv_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids[:self.max_len]
        else:
            input_ids = conv_text
        
        pixel_values = process_images([image], self.img_processor, self.model_config)[0].half()
        image_sizes = list(self.img_size)

        sample = {
            'input_ids': input_ids,
            'images': pixel_values,
            'image_sizes': image_sizes
        }
        
        return sample
    
    
class GroupRefDatasetHf(Dataset):
    def __init__(
            self,
            img_grouped_data,
            processor,
            img_str='<image>',
            n_point_range=(10, 30),
            img_dir='/home/ai2lab/work/datasets/',
            max_len=4096,
            img_size=(672,672),
            system_prompts=None,
            n_ref_per_img=5,
        ):
        self.img_grouped_data = img_grouped_data
        self.processor = processor
        self.img_str = img_str
        self.n_point_range = n_point_range
        self.img_dir = img_dir
        self.max_len = max_len
        self.img_size = img_size
        self.n_ref_per_img = n_ref_per_img
        self.system_prompts = system_prompts
        
        self.ref_data = []
        self.reload()

    def __len__(self):
        return len(self.ref_data)
    
    def reload(self):
        self.ref_data = []
        
        for img_path, refs in self.img_grouped_data.items():
            random.shuffle(refs)
            for i in range(0, len(refs), self.n_ref_per_img):
                self.ref_data.append(refs[i:i+self.n_ref_per_img])

    def __getitem__(self, idx, tokenize=True):
        ref_samples = self.ref_data[idx]
        image_path = ref_samples[0]['image_path']
        image = Image.open(self.img_dir + image_path).resize(self.img_size)

        conv = []
        if self.system_prompts is not None:
            conv.extend[{"role": "system", "content": random.choice(self.system_prompts)}]
        
        for i_conv, ref_sample in enumerate(ref_samples):

            s_phrase = random.choice(ref_sample['phrases'])
            answer_counts = ref_sample['answer_counts']
            question_box = f'{self.img_str}\n' if i_conv == 0 else ''
            question_box += f'Please provide the bounding box coordinate of the region this sentence describes ({answer_counts}):\n"{s_phrase}".'
            
            if len(ref_sample['bboxes']) == 0:
                answer_box = 'No object found.'
            else:
                answer_box = ' '.join([f'[{x[0]:03d},{x[1]:03d},{x[2]:03d},{x[3]:03d}]' for x in ref_sample['bboxes']])
            answer_box = ' ' + answer_box
            
            question_box = question_box + '\n' if question_box[-1] != '\n' else question_box
            # answer_box = answer_box + '\n' if answer_box[-1] != '\n' else answer_box
            conv.extend([
                {"role": "user", "content": question_box},
                {"role": "assistant", "content": answer_box}
            ])

            for bbox, p_n_ls in zip(ref_sample['bboxes'], ref_sample['points_and_labels']):
                n_sel_points = random.randint(self.n_point_range[0], self.n_point_range[1])
                sampled_points_and_labels = random.sample(p_n_ls, n_sel_points)
                
                points_txt = ' '.join([f'[{x[0]:03d},{x[1]:03d}]' for x in sampled_points_and_labels])
                question_points = 'Check if the points listed below are located on the object with coordinates [{:03d},{:03d},{:03d},{:03d}]:\n{}'.format(
                    bbox[0], bbox[1], bbox[2], bbox[3], points_txt)
                answer_points = '_' + ''.join(['Yes' if x[2] else 'No' for x in sampled_points_and_labels])
                
                # question_points = question_points + '\n' if question_points[-1] != '\n' else question_points
                # answer_points = answer_points + '\n' if answer_points[-1] != '\n' else answer_points
                conv.extend([
                    {"role": "user", "content": question_points},
                    {"role": "assistant", "content": answer_points}
                ])
            
        conv_text = self.processor.tokenizer.apply_chat_template(conv, tokenize=False)
        if conv_text.startswith('<s>'):
            conv_text = conv_text[3:]
        
        if tokenize:
            encoded = self.processor(conv_text, [image], return_tensors="pt")
            encoded['input_ids'] = encoded['input_ids'][0][:self.max_len]
            encoded['attention_mask'] = encoded['attention_mask'][0][:self.max_len]
            encoded['pixel_values'] = encoded['pixel_values'][0]
            encoded['image_sizes'] = encoded['image_sizes'][0]
        else:
            encoded = {
                'input_ids': conv_text,
                'images': [image],
                'image_sizes': [[image.size[0], image.size[1]]]
            }
        
        return encoded
    
    
    
class GroupRefDatasetHfV2(Dataset):
    def __init__(
            self,
            img_grouped_data,
            processor,
            img_str='<image>',
            n_point_range=(10, 30),
            img_dir='/home/ai2lab/work/datasets/',
            max_len=4096,
            img_size=(672,672),
            system_prompts=None,
            n_ref_per_img=5,
        ):
        self.img_grouped_data = img_grouped_data
        self.processor = processor
        self.img_str = img_str
        self.n_point_range = n_point_range
        self.img_dir = img_dir
        self.max_len = max_len
        self.img_size = img_size
        self.n_ref_per_img = n_ref_per_img
        self.system_prompts = system_prompts
        
        self.ref_data = []
        self.reload()

    def __len__(self):
        return len(self.ref_data)
    
    def reload(self):
        self.ref_data = []
        
        for img_path, refs in self.img_grouped_data.items():
            random.shuffle(refs)
            for i in range(0, len(refs), self.n_ref_per_img):
                self.ref_data.append(refs[i:i+self.n_ref_per_img])

    def __getitem__(self, idx, tokenize=True):
        ref_samples = self.ref_data[idx]
        image_path = ref_samples[0]['image_path']
        image = Image.open(self.img_dir + image_path).resize(self.img_size)

        conv = []
        if self.system_prompts is not None:
            conv.extend([{"role": "system", "content": random.choice(self.system_prompts)}])
        
        random.shuffle(ref_samples)
        is_over_max_len = False
        total_bbox = sum([len(x['bboxes']) for x in ref_samples])
        for i_conv, ref_sample in enumerate(ref_samples):
            
            if is_over_max_len:
                break

            s_phrase = random.choice(ref_sample['phrases'])
            answer_counts = ref_sample['answer_counts']
            question_box = f'{self.img_str}\n' if i_conv == 0 else ''
            question_box += f'Please provide the bounding box coordinate of the region this sentence describes ({answer_counts}):\n"{s_phrase}".'
            
            if len(ref_sample['bboxes']) == 0:
                answer_box = 'No object found.'
            else:
                answer_box = ' '.join([f'[{x[0]:03d},{x[1]:03d},{x[2]:03d},{x[3]:03d}]' for x in ref_sample['bboxes']])
                
            conv.extend([
                {"role": "user", "content": question_box},
                {"role": "assistant", "content": answer_box}
            ])
            conv_tokens = self.processor.tokenizer.apply_chat_template(conv, tokenize=True)
            if len(conv_tokens) > self.max_len:
                print('cut off')
                conv = conv[:-2]
                is_over_max_len = True
                break

            for bbox, p_n_ls in zip(ref_sample['bboxes'], ref_sample['points_and_labels']):
                n_sel_points = random.randint(self.n_point_range[0], self.n_point_range[1])
                if total_bbox <= 3:
                    n_sel_points = self.n_point_range[1]
                
                sampled_points_and_labels = random.sample(p_n_ls, n_sel_points)
                
                points_txt = ' '.join([f'[{x[0]:03d},{x[1]:03d}]' for x in sampled_points_and_labels])
                question_points = 'Check if the points listed below are located on the object with coordinates [{:03d},{:03d},{:03d},{:03d}]:\n{}'.format(
                    bbox[0], bbox[1], bbox[2], bbox[3], points_txt)
                answer_points = ''.join(['Yes' if x[2] else 'No' for x in sampled_points_and_labels])
                
                conv.extend([
                    {"role": "user", "content": question_points},
                    {"role": "assistant", "content": answer_points}
                ])
                conv_tokens = self.processor.tokenizer.apply_chat_template(conv, tokenize=True)
                if len(conv_tokens) > self.max_len:
                    print('cut off')
                    conv = conv[:-2]
                    is_over_max_len = True
                    break
            
        conv_text = self.processor.tokenizer.apply_chat_template(conv, tokenize=False)
        if conv_text.startswith('<s>'):
            conv_text = conv_text[3:]
        if conv_text.startswith('<|begin_of_text|>'):
            conv_text = conv_text[17:]
        
        if tokenize:
            encoded = self.processor(conv_text, [image], return_tensors="pt")
            encoded['input_ids'] = encoded['input_ids'][0][:self.max_len]
            encoded['attention_mask'] = encoded['attention_mask'][0][:self.max_len]
            encoded['pixel_values'] = encoded['pixel_values'][0].to(torch.bfloat16)
            encoded['image_sizes'] = encoded['image_sizes'][0]
        else:
            encoded = {
                'input_ids': conv_text,
                'images': [image],
                'image_sizes': [[image.size[0], image.size[1]]]
            }
        
        return encoded
    
    