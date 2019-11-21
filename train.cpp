


#include <Windows.h>
#include "cc_nb.h"
#include <iostream>
#include <pa_file\pa_file.h>
#include <highgui.h>
#include <fstream>
#include <mutex>
#include <thread>

using namespace cv;
using namespace std;
using namespace cc;
namespace L = cc::layers;



cc::Tensor vgg16conv(const cc::Tensor& input){
	cc::Tensor x = input;
	int num_output = 64;
	for (int i = 1; i <= 5; ++i){
		int n = i <= 2 ? 2 : 3;
		++n;
		for (int j = 1; j < n; ++j){
			x = L::conv2d(x, { 3, 3, num_output }, "same", { 1, 1 }, { 1, 1 }, cc::sformat("conv%d_%d", i, j));
			x = L::relu(x, cc::sformat("relu%d_%d", i, j));
		}
		x = L::max_pooling2d(x, { 2, 2, }, { 2, 2 }, { 0, 0 }, false, cc::sformat("pool%d", i));
		if (i < 4){
			num_output *= 2;
		}

		return x;
	}
}

cc::Tensor vgg16fc(const cc::Tensor& input){
	cc::Tensor x = input;
	for (int i = 6; i <= 7; ++i){
		x = L::dense(x, 4096, cc::sformat("fc%d", i), true);
		x = L::relu(x, cc::sformat("relu%d", i), true);
		x = L::dropout(x,0.5, cc::sformat("drop%d", i), true);
		}
	return x;
}

struct DataItem{
	std::shared_ptr<mutex> lock_;
	std::shared_ptr<Blob> top0;
	std::shared_ptr<Blob> top1;
};

class Dataset{
public:
	Dataset(){

		string dir = "D:/CC5.0-project/plugin/vgg-16/dataset";
		string labelfile = dir + "/labels.txt";

		map<string, int> labelmap;
		ifstream fin(labelfile, ios::in | ios::binary);
		string line;
		int lab = 0;	//0 is background
		while (getline(fin, line)){
			//if (line.back() == '\r' || line.back() == '\n')
			//	line.pop_back();

			string name = line;
			labelmap[name] = lab++;
		}
		numClass_ = lab;

		PaVfiles vfs;
		paFindFiles(dir.c_str(), vfs, "*.jpg", true);
		std::random_shuffle(vfs.begin(), vfs.end());

		int numtrain = vfs.size() * 0.9;
		for (int i = 0; i < vfs.size(); ++i){
			int p = vfs[i].rfind('\\');
			int np = vfs[i].rfind('\\', p - 1);
			string dirname = vfs[i].substr(np + 1, p - np - 1);

			if (i < numtrain)
				trainset.push_back(make_pair(vfs[i], labelmap[dirname]));
			else
				valset.push_back(make_pair(vfs[i], labelmap[dirname]));
		}
		printf("train: %d, val: %d\n", trainset.size(), valset.size());
	}

	int numClass(){
		return numClass_;
	}

	vector<pair<string, int>>& train(){
		return trainset;
	}

	vector<pair<string, int>>& val(){
		return valset;
	}

private:
	vector<pair<string, int>> trainset;
	vector<pair<string, int>> valset;
	int numClass_ = 0;
};

shared_ptr<Dataset> g_dataset;
#define trainbatch		32
#define valbatch		5

class MyData : public cc::BaseLayer{

public:
	SETUP_LAYERFUNC(MyData);

	MyData()
		:BaseLayer(), batchs(5){
	}

	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop){

		vector<pair<string, int>> dataset;
		if (phase == PhaseTest){
			batch_size = valbatch;
			dataset = g_dataset->val();
		}
		else{
			batch_size = trainbatch;
			dataset = g_dataset->train();
		}

		top[0]->reshape(batch_size, 3, 224, 224);
		top[1]->reshape(batch_size, 1);

		const int numcache = 15;
		allbatch.resize(numcache);
		for (int i = 0; i < numcache; ++i){
			allbatch[i].top0 = newBlob();
			allbatch[i].top1 = newBlob();
			allbatch[i].top0->reshapeLike(top[0]);
			allbatch[i].top1->reshapeLike(top[1]);
			allbatch[i].lock_.reset(new mutex());
		}
		
		datas.resize(dataset.size());

#pragma omp parallel for num_threads(32)
		for (int i = 0; i < dataset.size(); ++i){
			auto& item = dataset[i];
			Mat im = imread(item.first);
			int label = item.second;
			im.convertTo(im, CV_32F, 1 / 127.5, -1.0);
			datas[i] = make_pair(im, label);
		}

		thread([](Blob** top, int batch_size, vector<pair<Mat, int>>& datas, ThreadSafetyQueue<DataItem*>& batchs, int numcache, vector<DataItem>& allbatch){

			vector<int> inds;
			for (int i = 0; i < datas.size(); ++i)
				inds.push_back(i);
			std::random_shuffle(inds.begin(), inds.end());

			int cursor = 0;
			while (!batchs.eof()){
				for (int n = 0; n < numcache; ++n){
					std::unique_lock<mutex> l(*allbatch[n].lock_.get());
					Blob* top0 = allbatch[n].top0.get();
					Blob* top1 = allbatch[n].top1.get();
					for (int i = 0; i < batch_size; ++i){
						auto& item = datas[inds[cursor]];
						top0->setData(i, item.first);
						*(top1->mutable_cpu_data() + i) = item.second;

						cursor++;
						if (cursor == inds.size()){
							cursor = 0;
							std::random_shuffle(inds.begin(), inds.end());
						}
					}
					batchs.push(&allbatch[n]);
				}
			}
		}, top, batch_size, std::ref(datas), std::ref(batchs), numcache, std::ref(allbatch)).detach();
	}

	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop){

		DataItem* out = nullptr;
		while (!batchs.pull(out))
			std::this_thread::sleep_for(std::chrono::milliseconds(1));

		std::unique_lock<mutex> l(*out->lock_.get());
		top[0]->copyFrom(out->top0.get());
		top[1]->copyFrom(out->top1.get());
	}

private:
	int cursor = 0;
	vector<pair<Mat, int>> datas;
	int batch_size = 0;
	ThreadSafetyQueue<DataItem*> batchs;
	vector<DataItem> allbatch;
};

void main(){
	
	cc::installRegister();
	INSTALL_LAYER(MyData);

	g_dataset.reset(new Dataset());

	auto data = L::data("MyData", {"image", "label"}, "data");
	auto image = data[0];
	auto label = data[1];
	auto x = image;
	x = vgg16conv(x);
	x = L::max_pooling2d(x, { 2, 2 }, { 2, 2 }, { 0, 0 }, false, "pool5");
	//x = vgg16fc(x);
	x = L::dense(x, g_dataset->numClass(), "fc8",true);
	   
	L::ODense* layer = (L::ODense*)x->owner.get();
	layer->weight_initializer.reset(new cc::Initializer());
	layer->weight_initializer->type = "xavier";
	layer->bias_initializer.reset(new cc::Initializer());
	layer->bias_initializer->type = "constant";
	layer->bias_initializer->value = 0;

	auto loss = cc::loss::softmax_cross_entropy(x, label, "loss");
	auto accuracy = cc::metric::classifyAccuracy(x, label, "accuracy");

	auto op = cc::optimizer::momentumStochasticGradientDescent(cc::learningrate::step(0.001, 0.1, 3000), 0.9);
	op->max_iter = 2000;
	op->display = 10;
	op->snapshot = 100;
	op->test_interval = (int)(ceil(g_dataset->train().size() / (float)trainbatch));
	op->test_iter = (int)(ceil(g_dataset->val().size() / (float)valbatch));
	op->weight_decay = 0.0002f;
	//op->test_initialization = false;
	op->device_ids = { 0 };
	op->snapshot_prefix = "model_";
	op->minimize({ loss, accuracy });
	//op->minimizeFromFile("a.prototxt");
	printf("%s\n", op->seril().c_str());

	auto graph = engine::caffe::buildGraph({ loss, accuracy });
	if (cc::plugin::postPrototxt("netdebug", "net", graph.c_str(), graph.length())){
		cc::plugin::openNetscope("netdebug");
	}
	cc::train::caffe::run(op, [](OThreadContextSession* session, int step, float smoothed_loss){

	});
}