#ifndef  __MYLAYER_HPP__
#define __MYLAYER_HPP__
#include <iostream>
#include <memory>
#include "myBlob.hpp"

using std::vector;
using std::shared_ptr;

// 每一层的参数
struct Param
{
	// 卷积
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;
	string conv_weight_init;

	// 池化层
	int pool_stride;
	int pool_width;
	int pool_height;

	// 全连接层
	int fc_kernels;
	string fc_weight_init;

	// dropout的比例
	double drop_rate;

    // relu函数的上限阈值
	double th;
};

// 各个层都会继承Layer类的
class Layer
{
public:
	Layer(){}
	virtual ~Layer(){}
	virtual void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param) = 0;
	virtual void calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param) = 0;
	virtual void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode) = 0;
	virtual void backward(const shared_ptr<Blob>& din, 
											const vector<shared_ptr<Blob>>& cache, 
											vector<shared_ptr<Blob>>& grads,
											const Param& param) = 0;
};

class ConvLayer : public Layer
{
public:
	ConvLayer(){}
	~ConvLayer(){}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din,
								const vector<shared_ptr<Blob>>& cache,
								vector<shared_ptr<Blob>>& grads,
								const Param& param);
};

class ReluLayer : public Layer
{
public:
	ReluLayer(){}
	~ReluLayer(){}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din,
								const vector<shared_ptr<Blob>>& cache,
								vector<shared_ptr<Blob>>& grads,
								const Param& param);
};


class TanhLayer : public Layer
{
public:
	TanhLayer(){}
	~TanhLayer(){}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din,
								const vector<shared_ptr<Blob>>& cache,
								vector<shared_ptr<Blob>>& grads,
								const Param& param);
};

class PoolLayer : public Layer
{
public:
	PoolLayer(){}
	~PoolLayer(){}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din,
								const vector<shared_ptr<Blob>>& cache,
								vector<shared_ptr<Blob>>& grads,
								const Param& param);
};

class FcLayer : public Layer
{
public:
	FcLayer(){}
	~FcLayer(){}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din,
								const vector<shared_ptr<Blob>>& cache,
								vector<shared_ptr<Blob>>& grads,
								const Param& param);
};

class DropoutLayer : public Layer
{
public:
	DropoutLayer(){}
	~DropoutLayer(){}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din,
								const vector<shared_ptr<Blob>>& cache,
								vector<shared_ptr<Blob>>& grads,
								const Param& param);
private:
	shared_ptr<Blob> drop_mask;
};
//--------------------------------------------------------------------------------------------------------------------
class BNLayer : public Layer   
{
public:
	BNLayer() : running_mean_std_init(false){}
	~BNLayer() {}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din,
								const vector<shared_ptr<Blob>>& cache,
								vector<shared_ptr<Blob>>& grads,
								const Param& param);
private:
	bool running_mean_std_init;
	// 均值、方差、标准差
	shared_ptr<cube> mean_;
	shared_ptr<cube> var_;
	shared_ptr<cube> std_;
};

class ScaleLayer : public Layer 
{
public:
	ScaleLayer() {}
	~ScaleLayer() {}
	void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const Param& param);
	void calcShape(const vector<int>&inShape, vector<int>&outShape, const Param& param);
	void forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const Param& param, string mode);
	void backward(const shared_ptr<Blob>& din,
								const vector<shared_ptr<Blob>>& cache,
								vector<shared_ptr<Blob>>& grads,
								const Param& param);
};

//--------------------------------------------------------------------------------------------------------------------
class SoftmaxLossLayer
{
public:
	static void softmax_cross_entropy_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout);
};

class SVMLossLayer
{
public:
	static void hinge_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout);
};








#endif  //__MYLAYER_HPP__