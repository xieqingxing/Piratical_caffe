#include "myNet.hpp"
#include <json/json.h>
#include <fstream>  
#include <cassert>  //#include <assert.h>  

using namespace std;

void NetParam::readNetParam(string file)
{
    // ��ȡѵ����صĲ������������
	ifstream ifs;
	ifs.open(file);
	assert(ifs.is_open());
	Json::Reader reader;
	Json::Value value; // �洢��
	if (reader.parse(ifs, value))
	{
	    // ����ѵ������
		if (!value["train"].isNull())
		{
		    // ���value["train"]�е����в���
		    // Ȼ������ת���ʱ��洢
			auto &tparam = value["train"];
			this->lr = tparam["learning rate"].asDouble();
			this->lr_decay = tparam["lr decay"].asDouble();
			this->optimizer = tparam["optimizer"].asString();
			this->momentum = tparam["momentum coefficient"].asDouble();
			this->reg = tparam["reg coefficient"].asDouble();
			this->rms_decay = tparam["rmsprop decay"].asDouble();
			this->num_epochs = tparam["num epochs"].asInt();
			this->batch_size = tparam["batch size"].asInt();
			this->eval_interval = tparam["evaluate interval"].asInt();
			this->lr_update = tparam["lr update"].asBool();
			this->snap_shot = tparam["snapshot"].asBool();
			this->snapshot_interval = tparam["snapshot interval"].asInt();
			this->fine_tune = tparam["fine tune"].asBool();
			this->preTrainModel = tparam["pre train model"].asString();
		}
		// �����������
		if (!value["net"].isNull())
		{
			auto &nparam = value["net"];
			// ��������
			for (int i = 0; i < (int)nparam.size(); ++i)
			{
				auto &ii = nparam[i];
				// �������Ƿ��ظ�
				if (0 == count(layers.begin(), layers.end(), ii["name"].asString()))
                {
					this->layers.push_back(ii["name"].asString());
				}
				else
				{
					cerr << "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n   There is a Repeated layer name in Your json File ,  Please check: " << ii["name"].asString() << "\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" << endl;
				}
				this->ltypes.push_back(ii["type"].asString());
				// ���ݲ����ͼ��뵽lparams��
				if (ii["type"].asString() == "Conv")
				{
					this->lparams[ii["name"].asString()].conv_stride = ii["stride"].asInt();
					this->lparams[ii["name"].asString()].conv_kernels = ii["kernel num"].asInt();
					this->lparams[ii["name"].asString()].conv_pad = ii["pad"].asInt();
					this->lparams[ii["name"].asString()].conv_width = ii["kernel width"].asInt();
					this->lparams[ii["name"].asString()].conv_height = ii["kernel height"].asInt();
					this->lparams[ii["name"].asString()].conv_weight_init = ii["conv weight init"].asString();
				}
				if (ii["type"].asString() == "Pool")
				{
					this->lparams[ii["name"].asString()].pool_stride = ii["stride"].asInt();
					this->lparams[ii["name"].asString()].pool_width = ii["kernel width"].asInt();
					this->lparams[ii["name"].asString()].pool_height = ii["kernel height"].asInt();
				}
				if (ii["type"].asString() == "Fc")
				{
					this->lparams[ii["name"].asString()].fc_kernels = ii["kernel num"].asInt();
					this->lparams[ii["name"].asString()].fc_weight_init = ii["fc weight init"].asString();
				}
				if (ii["type"].asString() == "Dropout")
				{
					this->lparams[ii["name"].asString()].drop_rate = ii["drop rate"].asDouble();
				}
                if (ii["type"].asString() == "Relu")
				{
					this->lparams[ii["name"].asString()].th = ii["threshold"].asDouble();    //����Relu�������Ͻ���ֵ
				}
			}
		}
	}
	else   //json�ļ������﷨���󣡣���
		cerr << "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n         There is a synax error in Your json File ,  Please check !!!   \n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" << endl;
}



void Net::initNet(NetParam& param, vector<shared_ptr<Blob>>& X, vector<shared_ptr<Blob>>& Y)
{
    // ����
	layers_ = param.layers;
	// ������
	ltypes_ = param.ltypes;
	for (int i = 0; i < layers_.size(); ++i)
	{
		cout << "layer = " << layers_[i] << " ; " << "ltype = " << ltypes_[i] << endl;
	}

	//2.��ʼ��Net����س�Ա����
    X_train_ = X[0];
    Y_train_ = Y[0];
    X_val_ = X[1];
    Y_val_ = Y[1];

    // ��ʼ������ṹ
	for (int i = 0; i < (int)layers_.size(); ++i)
	{
		data_[layers_[i]] = vector<shared_ptr<Blob>>(3, NULL);
        diff_[layers_[i]] = vector<shared_ptr<Blob>>(3, NULL);
        step_cache_[layers_[i]] = vector<shared_ptr<Blob>>(3, NULL);
        outShapes_[layers_[i]] = vector<int>(4);
	}

	// ���ÿһ���w,b�ĳ�ʼ�������һ����softmax���ó�ʼ��
	shared_ptr<Layer> myLayer(NULL);
	vector<int> inShape = { param.batch_size,
												X_train_->get_C(),
												X_train_->get_H(),
												X_train_->get_W() };

	for (int i = 0; i < (int)layers_.size()-1; ++i)
	{
		string lname = layers_[i];
		string ltype = ltypes_[i];
		if (ltype == "Conv")
		{
			myLayer.reset(new ConvLayer);	
		}
		if (ltype == "Relu")
		{
			myLayer.reset(new ReluLayer);
		}
		if (ltype == "Tanh")
		{
			myLayer.reset(new TanhLayer);
		}
		if (ltype == "Pool")
		{
			myLayer.reset(new PoolLayer);
		}
		if (ltype == "Fc")
		{
			myLayer.reset(new FcLayer);
		}
		if (ltype == "Dropout")
		{
			myLayer.reset(new DropoutLayer);
		}
		if (ltype == "BN")
		{
			myLayer.reset(new BNLayer);
		}
		if (ltype == "Scale")
		{
			myLayer.reset(new ScaleLayer);
		}
		myLayers_[lname] = myLayer;
        myLayer->initLayer(inShape, lname, data_[lname], param.lparams[lname]);
		myLayer->calcShape(inShape, outShapes_[lname], param.lparams[lname]);
		inShape.assign(outShapes_[lname].begin(), outShapes_[lname].end());
	}

	// ����finetune�ķ���
	if (param.fine_tune)
	{
		fstream input(param.preTrainModel, ios::in | ios::binary);
		if (!input)
		{
			cout << param.preTrainModel << " was not found ������" << endl;
			return;
		}

		shared_ptr<piratical_caffe::snapshotModel>  snapshot_model(new gordon::snapshotModel);
		// protobuf�ṩ�ķ���
		if (!snapshot_model->ParseFromIstream(&input))
		{
			cout<< "Failed to parse the " << param.preTrainModel << " ������" << endl;
			return;
		}
		cout << "--- Load the" << param.preTrainModel << " sucessful" << endl;
        // ����ģ�Ͳ���
		loadModelParam(snapshot_model);
	}



}

void Net::trainNet(NetParam& param)
{
	int N = X_train_->get_N();
	cout << "N = " << N << endl;
	int iter_per_epoch = N / param.batch_size;
	//�ܵ�������
	int num_batchs = iter_per_epoch * param.num_epochs;
	cout << "num_batchs(iterations) = " << num_batchs << endl;

	for (int iter = 0; iter < num_batchs; ++iter)
	{
		// ��ȡһ��mini-batch
		shared_ptr<Blob> X_batch;
		shared_ptr<Blob> Y_batch;
		X_batch.reset(new Blob(X_train_->subBlob((iter* param.batch_size) % N,((iter + 1)* param.batch_size) % N)));
		Y_batch.reset(new Blob(Y_train_->subBlob((iter* param.batch_size) % N,((iter + 1)* param.batch_size) % N)));

		train_with_batch(X_batch, Y_batch, param);

		// ����׼ȷ��
		if (iter%param.eval_interval == 0)
		{
			evaluate_with_batch(param);
			printf("iter_%d    lr: %0.6f    train_loss: %f    val_loss: %f    train_acc: %0.2f%%    val_acc: %0.2f%%\n",
				iter, param.lr, train_loss_, val_loss_, train_accu_ * 100, val_accu_ * 100);
		}


		// ����ģ�Ϳ��� https://blog.csdn.net/u011334621/article/details/51735418
		if (iter > 0 && param.snap_shot    &&     iter % param.snapshot_interval == 0)
		{
			// ��������ļ�outputFile
			char outputFile[40];   
			sprintf(outputFile, "./iter%d.piratical_caffe_model", iter);                      //sprintf()���Խ�����תΪ�ַ���
			fstream output(outputFile, ios::out | ios::trunc | ios::binary);     //�����ļ���������ɾ���ٴ����������ھ�ֱ�Ӵ�����

			// ��Blob�еĲ������浽��proto����ģ�snapshotModel������ݽṹ�У�
			shared_ptr<piratical_caffe::snapshotModel> snapshot_model(new piratical_caffe::snapshotModel);
			saveModelParam(snapshot_model);

			// д��������ļ�
			if (!snapshot_model->SerializeToOstream(&output))
			{
				cout << "Failed to Serialize snapshot_model To Ostream." << endl;
				return;
			}
		}
	}
}


void Net::train_with_batch(shared_ptr<Blob>&  X, shared_ptr<Blob>&  Y, NetParam& param,string mode)
{
	// ��䵽��ʼ��
	data_[layers_[0]][0]=X;
	data_[layers_.back()][1] = Y;

	// ǰ�򴫲�
	int n = layers_.size();
	for (int i = 0; i < n - 1; ++i)
	{
		string lname = layers_[i];
		shared_ptr<Blob> out;
		myLayers_[lname]->forward(data_[lname], out, param.lparams[lname],mode);
		data_[layers_[i+1]][0] = out;
	}

	// softmaxǰ����㣬���Ҽ���loss
	if (mode == "TRAIN")
	{
		if (ltypes_.back() == "Softmax")
			SoftmaxLossLayer::softmax_cross_entropy_with_logits(data_[layers_.back()], train_loss_, diff_[layers_.back()][0]);
		if (ltypes_.back() == "SVM")
			SVMLossLayer::hinge_with_logits(data_[layers_.back()], train_loss_, diff_[layers_.back()][0]);
	}
	else
	{
		if (ltypes_.back() == "Softmax")
			SoftmaxLossLayer::softmax_cross_entropy_with_logits(data_[layers_.back()], val_loss_, diff_[layers_.back()][0]);
		if (ltypes_.back() == "SVM")
			SVMLossLayer::hinge_with_logits(data_[layers_.back()], val_loss_, diff_[layers_.back()][0]);
	}
	
	// ���򴫲�
	if (mode == "TRAIN")
	{
		for (int i = n - 2; i >= 0; --i)
		{
			string lname = layers_[i];
			myLayers_[lname]->backward(diff_[layers_[i + 1]][0], data_[lname], diff_[lname], param.lparams[lname]);
		}
	}


	// l2����
	if (param.reg!=0)
		regular_with_batch(param,mode);

	// ��������
	if (mode == "TRAIN")
		optimizer_with_batch(param);

}

void Net::regular_with_batch(NetParam& param, string mode)
{
	int N = data_[layers_[0]][0]->get_N();  //��ȡ������������
	double reg_loss = 0;
	for (auto lname : layers_)
	{
		if (diff_[lname][1]) //ֻ�Դ�Ȩֵ�ݶȵĲ���д���
		{
			if (mode == "TRAIN")
				(*diff_[lname][1])  = (*diff_[lname][1]) + param.reg * (*data_[lname][1]) / N;
			reg_loss += accu(square((*data_[lname][1])));
		}
	}
	reg_loss = reg_loss* param.reg / (2 * N);
	if (mode == "TRAIN")
		train_loss_ = train_loss_ + reg_loss;
	else
		val_loss_ = val_loss_ + reg_loss;
}

void Net::optimizer_with_batch(NetParam& param)
{
	for (int i = 0; i < layers_.size() - 1; ++i)    //for lname in layers_
	{
		string lname = layers_[i];
		string ltype = ltypes_[i];
		// ����û��w��b�Ĳ�, ����BN�㣨��Ϊ�洢���ǲ���ʱ��ʹ�õĲ�����
		if (!data_[lname][1] || !data_[lname][2] || ltype == "BN")
			continue;

		for (int i = 1; i <= 2; ++i)
		{
            // ֧�������Ż���sgd/momentum/rmsprop
			assert(param.optimizer == "sgd" || param.optimizer == "momentum" || param.optimizer == "rmsprop");

			shared_ptr<Blob> dparam(new Blob(data_[lname][i]->size(),TZEROS));
			if (param.optimizer == "rmsprop")
			{
				double decay_rate = param.rms_decay;
				// ����ǵ�һ�θ��µĻ�����ʼ��Ϊ0
				if (!step_cache_[lname][i])
					step_cache_[lname][i].reset(new Blob(data_[lname][i]->size(), TZEROS));

				(*step_cache_[lname][i]) = decay_rate * (*step_cache_[lname][i]) + (1 - decay_rate)* (*diff_[lname][i]) * (*diff_[lname][i]);
				(*dparam) = -param.lr *  (*diff_[lname][i]) / sqrt((*step_cache_[lname][i]) + 1e-8);

			}
			else if (param.optimizer == "momentum")
			{
				if (!step_cache_[lname][i])
					step_cache_[lname][i].reset(new Blob(data_[lname][i]->size(), TZEROS));
				(*step_cache_[lname][i]) = param.momentum * (*step_cache_[lname][i]) + (*diff_[lname][i]);
				(*dparam) = -param.lr * (*step_cache_[lname][i]);
			}
			else
			{
				(*dparam) = -param.lr * (*diff_[lname][i]);
			}
			// ���ò��֣��ݶ��½�
			(*data_[lname][i]) = (*data_[lname][i]) + (*dparam);
		}
	}
	// ѧϰ��˥��
	if (param.lr_update)
		param.lr *= param.lr_decay;
}

void Net::evaluate_with_batch(NetParam& param)
{
	// ����ѵ����׼ȷ��
	shared_ptr<Blob> X_train_subset;  
	shared_ptr<Blob> Y_train_subset;
	int N = X_train_->get_N();
	if (N > 1000)
	{
        X_train_subset.reset(new Blob(X_train_->subBlob(0, 1000)));
        Y_train_subset.reset(new Blob(Y_train_->subBlob(0, 1000)));
	}
	else
	{
		X_train_subset = X_train_;
		Y_train_subset = Y_train_;
	}
	// testģʽ��ֻ����ǰ�򴫲�
	train_with_batch(X_train_subset, Y_train_subset, param,"TEST");
	train_accu_ =calc_accuracy(*data_[layers_.back()][1], *data_[layers_.back()][0]);


	train_with_batch(X_val_, Y_val_, param, "TEST");
	val_accu_ = calc_accuracy(*data_[layers_.back()][1], *data_[layers_.back()][0]);
}

double Net::calc_accuracy(Blob& Y, Blob& Predict)
{
	vector<int> size_Y = Y.size();
	vector<int> size_P = Predict.size();
	for (int i = 0; i < 4; ++i)
	{
		assert(size_Y[i] == size_P[i]);
	}
	// �ҳ���ǩֵY��Ԥ��ֵPredict���ֵ����λ�ý��бȽϣ���һ�£�����ȷ����+1
	int N = Y.get_N();  //��������
	int right_cnt = 0;  //��ȷ����
	for (int n = 0; n < N; ++n)
	{
		// �ο���ַ��http://arma.sourceforge.net/docs.html#index_min_and_index_max_member
		if (Y[n].index_max() == Predict[n].index_max())
			right_cnt++;
	}
    // ����׼ȷ��
	return (double)right_cnt / (double)N;
}

void Net::saveModelParam(shared_ptr<piratical_caffe::snapshotModel>& snapshot_model)
{
	for (auto lname : layers_)    //for lname in layers_
	{
		// ����û��w��b�Ĳ�
		if (!data_[lname][1] || !data_[lname][2])
		{
			continue;
		}

		for (int i = 1; i <= 2; ++i)
		{
            piratical_caffe::snapshotModel_paramBlok*  param_blok = snapshot_model->add_param_blok();   //����̬�����һ��paramBlock
			int N = data_[lname][i]->get_N();
			int C = data_[lname][i]->get_C();
			int H = data_[lname][i]->get_H();
			int W = data_[lname][i]->get_W();
			param_blok->set_kernel_n(N);
			param_blok->set_kernel_c(C);
			param_blok->set_kernel_h(H);
			param_blok->set_kernel_w(W);
			param_blok->set_layer_name(lname);
			if (i == 1)
			{
				param_blok->set_param_type("WEIGHT"); //д���������
			}
			else
			{
				param_blok->set_param_type("BIAS");
			}
			for (int n = 0; n<N; ++n)  
			{
				for (int c = 0; c < C; ++c) 
				{
					for (int h = 0; h<H; ++h)  
					{
						for (int w = 0; w<W; ++w)  
						{
                            piratical_caffe::snapshotModel_paramBlok_paramValue*  param_value = param_blok->add_param_value();   //����̬�����һ��paramValue
							param_value->set_value((*data_[lname][i])[n](h, w, c));
						}
					}
				}
			}

		}
	}
}

void Net::loadModelParam(const shared_ptr<piratical_caffe::snapshotModel>& snapshot_model)
{
	for (int i = 0; i < snapshot_model->param_blok_size(); ++i)  //���ȡ��ģ�Ϳ����еĵ�paramBlok���������Ƕ����Blob���ݽṹ��
	{	// ��snapshot_model��һȡ��paramBlok
		 const piratical_caffe::snapshotModel::paramBlok& param_blok = snapshot_model->param_blok(i);  //ȡ����ӦparamBlok
		// ȡ��paramBlok�еı���ͱ���
		string lname = param_blok.layer_name();
		string paramtype = param_blok.param_type();
		int N = param_blok.kernel_n();
		int C = param_blok.kernel_c();
		int H = param_blok.kernel_h();
		int W = param_blok.kernel_w();

		// ������ǰparamBlok�е�ÿһ��������ȡ�����������Ӧ��Blob��
		int val_idx = 0;
		shared_ptr<Blob> simple_blob(new Blob(N, C, H, W));  //�м�Blob
		for (int n = 0; n<N; ++n)
		{
			for (int c = 0; c < C; ++c)
			{
				for (int h = 0; h<H; ++h)
				{
					for (int w = 0; w<W; ++w)
					{
						const piratical_caffe::snapshotModel_paramBlok_paramValue& param_value = param_blok.param_value(val_idx);
						(*simple_blob)[n](h,w,c)=param_value.value();   //ȡ��ĳ������������Blob��Ӧλ�ã�
						val_idx++;  //param_blok��������������
					}
				}
			}
		}

		//4. ��simple_blob��ֵ��data_��
		if (paramtype == "WEIGHT")
			data_[lname][1] = simple_blob;
		else
			data_[lname][2] = simple_blob;
	}
}