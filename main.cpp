#include "myNet.hpp"
#include "myBlob.hpp"
#include <iostream>
#include <string>
#include <memory>

using namespace std;

int ReverseInt(int i)  //�Ѵ������ת��ΪС������
{
	unsigned char ch1, ch2, ch3, ch4;  //һ��int��4��char
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
		// mnistԭʼ�����ļ���32λ������ֵ�Ǵ�˴洢��C/C++������С�˴洢�����Զ�ȡ���ݵ�ʱ����Ҫ������д�С��ת��
		// magic_number�����������ļ�ʶ��ġ�
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);  //�ߵ��ֽڵ���
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

		// ��������ͼƬ��Ȼ��洢
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
    // �����Ͷ�ȡͼƬ�ĺ���һ��
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

	//1. ��60000��ͼƬ��59:1�ı�������Ϊѵ������59000�ţ�����֤����1000�ţ�
	shared_ptr<Blob> X_train(new Blob(X_tarin_ori->subBlob(0, 59000)));  //����ҿ����䣬��[ 0, 59000 )
	shared_ptr<Blob> Y_train(new Blob(Y_tarin_ori->subBlob(0, 59000)));
	shared_ptr<Blob> X_val(new Blob(X_tarin_ori->subBlob(59000, 60000)));
	shared_ptr<Blob> Y_val(new Blob(Y_tarin_ori->subBlob(59000, 60000)));

	vector<shared_ptr<Blob>> XX{ X_train, X_val };
	vector<shared_ptr<Blob>> YY{ Y_train, Y_val };

	//2. ��ʼ������ṹ
	Net myModel;
	myModel.initNet(net_param, XX, YY);

	//3. ��ʼѵ��
	cout << "------------ step3. Train start... ---------------" << endl;
	myModel.trainNet(net_param);
	cout << "------------ Train end... ---------------" << endl;
}

void trainModel_with_exVal(NetParam& net_param, shared_ptr<Blob> X_tarin_ori, shared_ptr<Blob> Y_tarin_ori,
																							shared_ptr<Blob> X_val_ori, shared_ptr<Blob> Y_val_ori)
{

    vector<shared_ptr<Blob>> XX{ X_tarin_ori, X_val_ori };
    vector<shared_ptr<Blob>> YY{ Y_tarin_ori, Y_val_ori };

	// ��ʼ������ṹ��ѵ��
	Net myModel;
	myModel.initNet(net_param, XX, YY);
	myModel.trainNet(net_param);
}

int main(int argc, char** argv)
{
	// ���������ѵ������
	string configFile = "./myModel_cnn.json";   //myModel_MLP      myModel_cnn     myModel
	NetParam net_param;
	net_param.readNetParam(configFile);

	// ��������Blob���󣬷ֱ����洢ͼƬ�ͱ�ǩ
	shared_ptr<Blob> images_train(new Blob(60000, 1, 28, 28, TZEROS));
	shared_ptr<Blob> labels_train(new Blob(60000, 10, 1, 1, TZEROS));

	//��ȡ����
	ReadMnistData("mnist_data/train/train-images.idx3-ubyte", images_train);
	ReadMnistLabel("mnist_data/train/train-labels.idx1-ubyte", labels_train);

	// ��ȡ��������
	shared_ptr<Blob> images_test(new Blob(10000, 1, 28, 28, TZEROS));
	shared_ptr<Blob> labels_test(new Blob(10000, 10, 1, 1, TZEROS));
	ReadMnistData("mnist_data/test/t10k-images.idx3-ubyte", images_test);
	ReadMnistLabel("mnist_data/test/t10k-labels.idx1-ubyte", labels_test);

	// ����ѵ�����Լ�
	int samples_num = 1000;
	shared_ptr<Blob> X_train(new Blob(images_train->subBlob(0, samples_num)));
	shared_ptr<Blob> Y_train(new Blob(labels_train->subBlob(0, samples_num)));
	shared_ptr<Blob> X_test(new Blob(images_test->subBlob(0, samples_num)));
	shared_ptr<Blob> Y_test(new Blob(labels_test->subBlob(0, samples_num)));
	trainModel_with_exVal(net_param, X_train, Y_train, X_test, Y_test);

	system("pause");
	return 0;
}

