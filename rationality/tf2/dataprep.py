# Extract (document,sentences) data
import os

from rationality import multi_sentence_tokens, MAX_SEQ_LEN

if __name__=="__main__":
 home_dir = os.environ['HOME']
 data_dir = os.path.join(home_dir, 'Data')
 # data_dir = '/blok/@data'
 guten_dir = os.path.join(data_dir, 'org', 'gutenberg', 'txt-files')

 print("doc_idx\ttokens")
 for doc_idx, doc_path, tokens in multi_sentence_tokens(guten_dir, max_len=MAX_SEQ_LEN, pad=None):
  print(f"{doc_idx}\t{doc_path}\t{' '.join(tokens)}")
