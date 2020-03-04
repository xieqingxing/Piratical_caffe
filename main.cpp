#include "myNet.hpp"
#include "myBlob.hpp"
#include <iostream>
#include <string>
#include <memory>

using namespace std;

int ReverseInt(int i)  //把大端数据转换为小端数据
{
	unsigned char ch1, ch2, ch3, ch4;  //一个int有4个char
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

//http://yann.lecun.com/exdb/mnist/
void ReadMnistData(string path, shared_ptr<Blob> &images)
{
	ifstream file(path, ios::binary);
	if (file.is_open())
	{
		// mnist原始数据文件中32位的整型值是大端存储，C/C++变量是小端存储，所以读取数据的时候，需要对其进行大小端转换
		// magic_number是用来进行文件识别的。
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);  //高低字节调换
		cout << "magic_number=" << magic_number << endl;
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		cout << "number_of_images=" << number_of_images << endl;
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		cout << "n_rows=" << n_rows << endl;
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		cout << "n_cols=" << n_cols << endl;

		// 遍历所有图片，然后存储
		for (int i = 0; i<number_of_images; ++i)
		{
			for (int h = 0; h<n_rows; ++h)
			{
				for (int w = 0; w<n_cols; ++w)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					(*images)[i](h, w, 0) = (double)temp / 255;
				}
			}
		}
	}
	else
	{
		cout << "no data file found :-(" << endl;
	}

}
void ReadMnistLabel(string path, shared_ptr<Blob> &labels)
{
    // 操作和读取图片的函数一样
	ifstream file(path, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		cout << "magic_number=" << magic_number << endl;
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		cout << "number_of_Labels=" << number_of_images << endl;
		for (int i = 0; i<number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			(*labels)[i](0, 0, (int)temp) = 1;
		}
	}
	else
	{
		cout << "no label file found :-(" << endl;
	}
}



void trainModel(NetParam& net_param, shared_ptr<Blob> X_tarin_ori, shared_ptr<Blob> Y_tarin_ori)
{

	//1. 将60000张图片以59:1的比例划分为训练集（59000张）和验证集（1000张）
	shared_ptr<Blob> X_train(new Blob(X_tarin_ori->subBlob(0, 59000)));  //左闭右开区间，即[ 0, 59000 )
	shared_ptr<Blob> Y_train(new Blob(Y_tarin_ori->subBlob(0, 59000)));
	shared_ptr<Blob> X_val(new Blob(X_tarin_ori->subBlob(59000, 60000)));
	shared_ptr<Blob> Y_val(new Blob(Y_tarin_ori->subBlob(59000, 60000)));

	vector<shared_ptr<Blob>> XX{ X_train, X_val };
	vector<shared_ptr<Blob>> YY{ Y_train, Y_val };

	//2. 初始化网络结构
	Net myModel;
	myModel.initNet(net_param, XX, YY);

	//3. 开始训练
	cout << "------------ step3. Train start... ---------------" << endl;
	myModel.trainNet(net_param);
	cout << "------------ Train end... ---------------" << endl;
}

void trainModel_with_exVal(NetParam& net_param, shared_ptr<Blob> X_tarin_ori, shared_ptr<Blob> Y_tarin_ori,
																							shared_ptr<Blob> X_val_ori, shared_ptr<Blob> Y_val_ori)
{

    vector<shared_ptr<Blob>> XX{ X_tarin_ori, X_val_ori };
    vector<shared_ptr<Blob>> YY{ Y_tarin_ori, Y_val_ori };

	// 初始化网络结构，训练
	Net myModel;
	myModel.initNet(net_param, XX, YY);
	myModel.trainNet(net_param);
}

int main(int argc, char** argv)
{
	// 解析网络和训练参数
	string configFile = "./myModel_cnn.json";   //myModel_MLP      myModel_cnn     myModel
	NetParam net_param;
	net_param.readNetParam(configFile);

	// 创建两个Blob对象，分别来存储图片和标签
	shared_ptr<Blob> images_train(new Blob(60000, 1, 28, 28, TZEROS));
	shared_ptr<Blob> labels_train(new Blob(60000, 10, 1, 1, TZEROS));

	//读取数据
	ReadMnistData("mnist_data/train/train-images.idx3-ubyte", images_train);
	ReadMnistLabel("mnist_data/train/train-labels.idx1-ubyte", labels_train);

	// 读取测试数据
	shared_ptr<Blob> images_test(new Blob(10000, 1, 28, 28, TZEROS));
	shared_ptr<Blob> labels_test(new Blob(10000, 10, 1, 1, TZEROS));
	ReadMnistData("mnist_data/test/t10k-images.idx3-ubyte", images_test);
	ReadMnistLabel("mnist_data/test/t10k-labels.idx1-ubyte", labels_test);

	// 划分训练测试集
	int samples_num = 1000;
	shared_ptr<Blob> X_train(new Blob(images_train->subBlob(0, samples_num)));
	shared_ptr<Blob> Y_train(new Blob(labels_train->subBlob(0, samples_num)));
	shared_ptr<Blob> X_test(new Blob(images_test->subBlob(0, samples_num)));
	shared_ptr<Blob> Y_test(new Blob(labels_test->subBlob(0, samples_num)));
	trainModel_with_exVal(net_param, X_train, Y_train, X_test, Y_test);

	system("pause");
	return 0;
}

