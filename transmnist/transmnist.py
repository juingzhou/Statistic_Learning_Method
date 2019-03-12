# 这是将MNIST中的数据集转换为csv格式的脚本
# 后续代码都会在csv文件的基础上进行编写, 这样大家看代码也能清除很多
# 代码由以下网址提供，表示感谢。
# https://pjreddie.com/projects/mnist-in-csv/
# 该py文件属于一个补充，不使用也不影响后续算法的实践。
# 转换后的CVS文件在Mnist文件夹中
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []
    
    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

if __name__ == '__main__':
    convert("transmnist.\Mnist\\t10k-images.idx3-ubyte", "transmnist.\Mnist\\t10k-labels.idx1-ubyte", "transmnist.\Mnist\\mnist_test.csv", 10000)
    convert("transmnist.\Mnist\\train-images.idx3-ubyte", "transmnist.\Mnist\\train-labels.idx1-ubyte", "transmnist.\Mnist\mnist_train.csv", 60000)