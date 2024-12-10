import fileinput

def split(i: int) -> str:
 x = i % 16

 if x == 0:
  return 'valid'

 if x == 1:
  return 'test'

 return 'train'

if __name__=="__main__":
 for i, line in enumerate(fileinput.input(openhook=fileinput.hook_compressed)):
  s = split(i)

  print(f"{s}\t{line}", end='')
  
