#ifndef __MYNET_HPP__
#define __MYNET_HPP__
#include "myLayer.hpp"
#include "myBlob.hpp"
#include "piratical_caffe.snapshotModel.pb.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>

using std::unordered_map;
using std::vector;
using std::string;
using std::shared_ptr;

struct NetParam      //c++�У�struct��class�÷�����һ�£���Ҫ�����Ǽ̳к�����ݷ���Ȩ�ޡ�
{
	/*ѧϰ��*/
	double lr;
	/*ѧϰ��˥��ϵ��*/
	double lr_decay;
	/*�Ż��㷨,:sgd/momentum/rmsprop*/
	string optimizer;
	/*momentumϵ�� */
	double momentum;
	/*rmsprop ˥��ϵ�� */
	double rms_decay;
	/*L2����ϵ�� */
	double reg;
	/*epoch���� */
	int num_epochs;
	/*ÿ������������*/
	int batch_size;
	/*ÿ������������������һ��׼ȷ�ʣ� */
	int eval_interval;
	/*�Ƿ����ѧϰ�ʣ�  true/false*/
	bool lr_update;
	/* �Ƿ񱣴�ģ�Ϳ��գ����ձ�����*/
	bool snap_shot;
	/*ÿ�������������ڱ���һ�ο��գ�*/ 
	int snapshot_interval;
	/* �Ƿ����fine-tune��ʽѵ��*/
	bool fine_tune;
	/*Ԥѵ��ģ���ļ�*/
	string preTrainModel;
	/*����*/
	vector <string> layers;
	/*������*/
	vector <string> ltypes;
	/*�����������*/
	unordered_map<string, Param> lparams;
	void readNetParam(string file);
};

class Net
{
public:
	void initNet(NetParam& param, vector<shared_ptr<Blob>>& X, vector<shared_ptr<Blob>>& Y);
	void trainNet(NetParam& param);
	void train_with_batch(shared_ptr<Blob>&  X, shared_ptr<Blob>&  Y, NetParam& param, string mode="TRAIN");
	void regular_with_batch(NetParam& param, string mode = "TRAIN");
	void optimizer_with_batch(NetParam& param);
	void evaluate_with_batch(NetParam& param);
	double calc_accuracy(Blob& Y, Blob& Predict);
	void saveModelParam(shared_ptr<piratical_caffe::snapshotModel>& snapshot_model);
	void loadModelParam(const shared_ptr<piratical_caffe::snapshotModel>& snapshot_model);
private:
	// ѵ����
	shared_ptr<Blob> X_train_;
	shared_ptr<Blob> Y_train_;
	// ��֤��
	shared_ptr<Blob> X_val_;
	shared_ptr<Blob> Y_val_;

	vector<string> layers_;  //����
	vector<string> ltypes_; //������
	double train_loss_;
	double val_loss_;
	double train_accu_;
	double val_accu_;
	//
	unordered_map<string, vector<shared_ptr<Blob>>> data_;    //ǰ�������Ҫ�õ���Blob data_[0]=X,  data_[1]=W,data_[2] = b;
	unordered_map<string, vector<shared_ptr<Blob>>> diff_;    //���������Ҫ�õ���Blob diff_[0]=dX,  diff_[1]=dW,diff_[2] = db;
	unordered_map<string, vector<shared_ptr<Blob>>> step_cache_;   //�洢�ۼ��ݶȣ�momentum��rmsprop��
	unordered_map<string, shared_ptr<Layer>> myLayers_;
	unordered_map<string,vector<int>> outShapes_;    //�洢ÿһ�������ߴ�
};

#endif