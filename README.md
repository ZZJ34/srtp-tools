# selection.py

## 数据集有用标注的筛选

```
--pictureDir    -p   图片文件夹绝对路径(末尾没有/)
--labelDir      -l   标注文件夹绝对路径(末尾没有/)
--modelSize     -m   模型输入大小
--threshold     -t   阈值大小
--outputDir     -o   输出绝对路径(末尾没有/)
--readFromLine  -r   从标注文件的第几行读取


会产生两个输出文件
abandon.csv => 无用标注
useful.csv  => 有用标注


也可以在图片标出来目标，取消注释就好


样例执行代码
python3 ./selection.py 
-p /Volumes/ZZJ/test/picture 
-l /Volumes/ZZJ/test/label 
-m 224 
-t 10 
-o /Volumes/ZZJ/test/output 
-r 3
```

# infer.py 

## 图片分割代码(不是我写的)

# reasoning.py 

## 模型推理
