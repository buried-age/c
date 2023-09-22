
import argparse
import os
import sys

import dill
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from dataset import LEBertDataset
from model import LEBertSequenceTagging

sys.path.append("/home/ubuntu/sequence_tagging")
from tools.model_save import save_module
from tools import log_util
logger = log_util.get_logger(name="train", file="model.log", level="INFO")


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # 数据准备
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    special_tokens_dict = {
        "additional_special_tokens": ["<MCD>", "<AMC>", "<ATP>", "<ECD>", "<AEC>", "<CSE>", "<FAC>", "<SYS>", "<COM>",
                                      "<ELE>", "<DTD>", "<ELF>", "<RDN>", "<IOE>", "<OMN>", "<MMF>", "<CLE>", "<MAI>",
                                      "<CAR>", "<IMP>", "<ADJ>", "<INS>", "<ALA>", "<EBN>", "<TER>", "<URG>", "<EAR>"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    if args.train_pkl_path:
        with open(args.train_pkl_path, "rb") as f:
            dataset = dill.load(f)
    else:
        dataset = LEBertDataset.load_from_path(data_path=args.train_data_path,
                                               slot_path=args.slot_label_path,
                                               tokenizer=tokenizer)
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn)

    # 模型准备
    model = LEBertSequenceTagging.from_pretrained(args.model_path,
                                                  slot_num=dataset.tags_num,
                                                  token_size=len(tokenizer),
                                                  ignore_mismatched_sizes=True)
    model = model.to(device)

    # 训练配置
    if args.max_training_steps > 0:
        total_steps = args.max_training_steps
    else:
        total_steps = len(dataset) * args.train_epochs // args.gradient_accumulation_steps // args.batch_size

    parameter_names_no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [
            para for para_name, para in model.named_parameters()
            if not any(nd_name in para_name for nd_name in parameter_names_no_decay)
        ],
            'weight_decay': args.weight_decay},
        {'params': [
            para for para_name, para in model.named_parameters()
            if any(nd_name in para_name for nd_name in parameter_names_no_decay)
        ],
            'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    update_steps = 0
    total_loss = 0.
    for epoch in range(args.train_epochs):
        model.train()
        step = 0
        for batch in dataloader:
            step += 1
            input_ids, slot_ids, position_ids = batch
            outputs = model(input_ids=torch.tensor(input_ids).long().to(device),
                            position_ids=torch.tensor(position_ids).long().to(device),
                            tags=torch.tensor(slot_ids).long().to(device))

            loss = outputs['loss']
            total_loss += loss

            # 梯度累计
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                update_steps += 1

                logger.info("total step {} epoch {} : loss {}".format(update_steps, epoch, loss))
                if args.saving_steps > 0 and update_steps % args.saving_steps == 0:
                    save_module(model, args.save_dir, module_name='model',
                                additional_name="model_step{}".format(update_steps))

        if args.saving_epochs > 0 and (epoch + 1) % args.saving_epochs == 0:
            save_module(model, args.save_dir, module_name='model', additional_name="model_epoch_{}".format(epoch))

        if update_steps > total_steps:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # environment parameters
    parser.add_argument("--cuda_devices", type=str, default='0', help='set cuda device numbers')
    parser.add_argument("--no_cuda", action='store_true', default=False, help='whether use cuda device for training')

    # model parameters
    parser.add_argument("--tokenizer_path", type=str, default='hfl/chinese-roberta-wwm-ext',
                        help="pretrained tokenizer loading path")
    parser.add_argument("--model_path", type=str, default='hfl/chinese-roberta-wwm-ext',
                        help="pretrained model loading path")

    # data parameters
    parser.add_argument("--train_data_path", type=str, default='../datasets/industrial/train.json',
                        help="training data path")
    parser.add_argument("--train_pkl_path", type=str, default='../datasets/industrial/train.pkl',
                        help="training data path")

    # training parameters
    parser.add_argument("--save_dir", type=str, default='saved_model', help="directory to save the model")
    parser.add_argument("--max_training_steps", type=int, default=0,
                        help='max training step for optimizer, if larger than 0')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="number of updates steps to accumulate before performing a backward() pass.")
    parser.add_argument("--saving_steps", type=int, default=1000, help="parameter update step number to save model")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="parameter update step number to print logging info.")
    parser.add_argument("--saving_epochs", type=int, default=5, help="parameter update epoch number to save model")

    parser.add_argument("--batch_size", type=int, default=64, help='training data batch size')
    parser.add_argument("--train_epochs", type=int, default=500, help='training epoch number')

    parser.add_argument("--learning_rate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="epsilon for Adam optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup step number")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="maximum norm for gradients")

    args = parser.parse_args()

    train(args)
