# -*- coding: utf-8 -*-
import os
import sys
import re
import pandas as pd
import json
from time import time
import argparse

from datasets import Dataset

from tokenizers import models,normalizers, pre_tokenizers,processors, decoders
from tokenizers import trainers, Tokenizer #for training and selecting configurations
from tokenizers import SentencePieceBPETokenizer,SentencePieceUnigramTokenizer
from transformers import AlbertTokenizerFast,GPT2TokenizerFast,BertTokenizerFast,PreTrainedTokenizerFast,AutoTokenizer

class TigrignaTokenizer():

    def __init__(self,tokenizer_name,vocab_size=32000,special_tokens=[],max_length=512):
        self.vocab_size=vocab_size
        self.special_tokens=special_tokens
        self.max_length=max_length
        self.training_time=0.0

        # Defensive handling: allow tokenizer_name to be None (e.g. when no CLI arg
        # was supplied). Use 'Unigram' as a sensible default.
        if tokenizer_name is None:
            tokenizer_name = 'Unigram'

        # Normalize to string for safe lower()/startswith() calls
        _tname = str(tokenizer_name)

        if _tname == "Unigram" or _tname.lower().startswith("u"):
            self.tokenizer=Tokenizer(models.Unigram())
            self.trainer=trainers.UnigramTrainer(vocab_size=self.vocab_size,special_tokens=self.special_tokens,unk_token="<ለየለ>")
            self.tokenizer.pre_tokenizer=pre_tokenizers.Metaspace()
            self.tokenizer_name = 'Unigram'
        
        elif _tname=="BPE" or _tname.lower().startswith("b"):
            self.tokenizer=Tokenizer(models.BPE())
            self.trainer=trainers.BpeTrainer(vocab_size=self.vocab_size,special_tokens=self.special_tokens,unk_token="<ለየለ>")
            self.tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel()
            self.tokenizer_name = 'BPE'
        
        elif _tname=="WordPiece" or _tname.lower().startswith("w"):
            self.tokenizer=Tokenizer(models.WordPiece())
            self.trainer=trainers.WordPieceTrainer(vocab_size=self.vocab_size,special_tokens=self.special_tokens,unk_token="<ለየለ>")
            self.tokenizer.pre_tokenizer=pre_tokenizers.BertPreTokenizer()
            self.tokenizer_name = 'WordPiece'
        
        self.tokenizer.normalizer=normalizers.Sequence([normalizers.NFD(),
                                normalizers.Replace("[MASK]","[ሽፉን]"),
                                normalizers.Replace("<mask>","[ሽፉን]"),
                                normalizers.Replace("<unk>","<ለየለ>"),
                                normalizers.Replace("e.g","ዓ.ም")])
            
    
    def add_special_token(self,tokenizer_tr): 
    #tokenizer=PreTrainedTokenizerFast(tokenizer_object=tokenizer_tr,model_max_length=max_length, special_tokens=special_tokens)
    
        self.tokenizer.bos_token = "[ጀመረ]"
        self.tokenizer.bos_token_id = tokenizer_tr.token_to_id("[ከፋሊ]")
        self.tokenizer.pad_token = "<መልእ>"
        self.tokenizer.pad_token_id =  tokenizer_tr.token_to_id("<መልእ>")
        self.tokenizer.eos_token = "[ከፋሊ]"
        self.tokenizer.eos_token_id =  tokenizer_tr.token_to_id("[ከፋሊ]")
        self.tokenizer.unk_token = "[ለየለ]"
        self.tokenizer.unk_token_id = tokenizer_tr.token_to_id("[ለየለ]")
        self.tokenizer.cls_token = "[ጀመረ]"
        self.tokenizer.cls_token_id =  tokenizer_tr.token_to_id("[ጀመረ]")
        self.tokenizer.sep_token = "[ከፋሊ]"
        self.tokenizer.sep_token_id =  tokenizer_tr.token_to_id("[ከፋሊ]")
        self.tokenizer.mask_token = "[ሽፉን]"
        self.tokenizer.mask_token_id = tokenizer_tr.token_to_id("[ሽፉን]")

        return self.tokenizer
    
    def batch_iterator(self,dataset,batch_size=64):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]
    
    def train_tokenizer(self,dataset):
        strt=time()
        self.tokenizer.train_from_iterator(self.batch_iterator(dataset),trainer=self.trainer)
        print("Training FINISHED!!",(time()-strt))
        self.training_time=time()-strt
        
        self.tokenizer.add_special_tokens(self.special_tokens)
        cls_token_id = self.tokenizer.token_to_id("[ጀመረ]")
        sep_token_id = self.tokenizer.token_to_id("[ከፋሊ]")

        #print(cls_token_id,sep_token_id)
        self.tokenizer.post_processor = processors.TemplateProcessing(single="[CLS]:0 $A:0 [SEP]:0",
                                                                pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
                                                                special_tokens=[("[CLS]", cls_token_id),("[SEP]", sep_token_id),],)
        
        if self.tokenizer_name == 'Unigram':
            self.tokenizer = AlbertTokenizerFast(tokenizer_object=self.tokenizer)
        elif self.tokenizer_name =="BPE":
            self.tokenizer = GPT2TokenizerFast(tokenizer_object=self.tokenizer)
        elif self.tokenizer_name == 'WordPiece':
            self.tokenizer = BertTokenizerFast(tokenizer_object=self.tokenizer)
        
        return self.tokenizer
    
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Tigrigna tokenizer argument parser")

    # Add arguments
    parser.add_argument("--tokenizer_name", type=str, help="The name of the tokenizer", default="Unigram")
    parser.add_argument("--vocab_size", type=int, help="The size of vocaublary of the tokenizer", default=32000)
    parser.add_argument("--data_path", type=str, help="Path to a single file or a directory containing text files. Defaults to the repository tig_dataset directory.", default="../tig_dataset")
    #parser.add_argument("--special_tokens", type=list, help="The list of special tokens for the tokenizer")

    # Parse arguments
    args = parser.parse_args()
    vocab_size=args.vocab_size#["vocab_size"]
    tokenizer_name=args.tokenizer_name#["tokenizer_name"]
    special_tokens=['[ጀመረ]', '[ከፋሊ]', '[መልእ]', '[ለየለ]', '[ሽፉን]']#args.special_tokens#["special_tokens"]

    tokenizer_tig=TigrignaTokenizer(tokenizer_name=tokenizer_name,vocab_size=vocab_size,special_tokens=special_tokens)


    # Allow the user to pass either a single file or a directory of files.
    data_path = args.data_path
    data_path = os.path.abspath(os.path.expanduser(data_path))

    all_lines = []
    if os.path.isfile(data_path):
        # Single file: read lines
        try:
            with open(data_path, "r", encoding="utf-8") as file:
                for line in file:
                    s = line.strip()
                    if s:
                        all_lines.append(s)
        except Exception as e:
            print(f"Error reading file '{data_path}': {e}")
            sys.exit(1)
    elif os.path.isdir(data_path):
        # Directory: iterate over files
        for file_name in os.listdir(data_path):
            file_path = os.path.join(data_path, file_name)
            if not os.path.isfile(file_path):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    for line in file:
                        s = line.strip()
                        if s:
                            all_lines.append(s)
            except Exception as e:
                print(f"Warning: failed to read '{file_path}': {e}")
                continue
    else:
        print(f"The specified data_path does not exist or is not a file/directory: '{data_path}'")
        sys.exit(1)

    tig_dataset={"text":all_lines}
    tig_dataset=Dataset.from_dict(tig_dataset)
    print("Datasets ready! Sentences: ",len(all_lines))

    #tokenizer=tokenizer_tig.train_tokenizer(dataset=tig_dataset)

    #tokenizer.save_pretrained("/home/aberhe/Projects/SANTAL/Course/data/tokenizers/Tig_"+tokenizer_name+"_"+str(vocab_size))

if __name__ == "__main__":
    main()