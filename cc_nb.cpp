

#ifdef WIN32
#include <Windows.h>
#endif


#include "cc_nb.h"
#include <stdarg.h>
#include <thread>
#include <mutex>
#include <map>
#include <stack>


using namespace std;
using namespace cc;
using namespace cv;


namespace cc{

	///////////////////////////////////////////////////////////////
	string sformat(const char* fmt, ...){
		va_list vl;
		va_start(vl, fmt);

		char buffer[10000];
		vsprintf(buffer, fmt, vl);
		return buffer;
	}

	struct OThreadContextSessionImpl : public OThreadContextSession{

		std::function<StepEndCallbackFunctional> step_end_callback_;
		cc::Solver* solver_ = nullptr;
		string solver_pb;
		string net_pb;
		std::thread::id thread_id_;
		stack<string> namescope_stack_;
		map<std::string, int> layers_last_name_number_map_;
		map<std::string, void*> value_store_map_;
		LayerID next_layer_id_ = 1;

		static mutex global_lock_;
		static map<std::thread::id, std::shared_ptr<OThreadContextSessionImpl>> global_session_pool_;

		static OThreadContextSessionImpl* getSession(){
			std::thread::id tid = std::this_thread::get_id();
			std::unique_lock<mutex> l(OThreadContextSessionImpl::global_lock_);

			OThreadContextSessionImpl* session = nullptr;
			if (OThreadContextSessionImpl::global_session_pool_.find(tid) ==
				OThreadContextSessionImpl::global_session_pool_.end()){

				session = new OThreadContextSessionImpl();
				session->thread_id_ = tid;
				OThreadContextSessionImpl::global_session_pool_[tid].reset(session);
			}
			else{
				session = OThreadContextSessionImpl::global_session_pool_[tid].get();
			}
			return session;
		}

		virtual cc::Blob* get_tensor_blob(const char* blob_name){
			if (this->solver_){
				if (this->solver_->net()){
					return this->solver_->net()->blob(blob_name);
				}
			}
			return nullptr;
		}

		//获取存储在session中的值
		virtual void* get(const char* key){
			auto itr = value_store_map_.find(key);
			if (itr == value_store_map_.end())
				return nullptr;
			return itr->second;
		}

		virtual void put(const char* key, void* value){
			value_store_map_[key] = value;
		}

		virtual LayerID next_layer_id(){
			return next_layer_id_++;
		}
	};

	mutex OThreadContextSessionImpl::global_lock_;
	map<std::thread::id, std::shared_ptr<OThreadContextSessionImpl>> OThreadContextSessionImpl::global_session_pool_;

	OThreadContextSession* OThreadContextSession::getSession(){
		return OThreadContextSessionImpl::getSession();
	}

	//
	//    计算标准的DNN相关输出shape
	//
	int compute_std_dnn_output_shape(int input_dim, int kernel_dim, int strides, int padding, int dilation){
		int kernel_extent = dilation * (kernel_dim - 1) + 1;
		return (input_dim + 2 * padding - kernel_extent) / strides + 1;
	}

	//
	//    获取名称，基于当前上下文中的scope指定名称
	//    返回名称以： scope / name 的形式，若scope为空，则返回name
	//
	string get_name_with_scope(const string& name){
		if (OThreadContextSessionImpl::getSession()->namescope_stack_.empty())
			return name;

		return OThreadContextSessionImpl::getSession()->namescope_stack_.top() + "/" + name;
	}

	//
	//    scope的具体实现定义，构造的时候push，析构的时候pop
	//
	name_scope::name_scope(const string& name){
		OThreadContextSessionImpl::getSession()->namescope_stack_.push(get_name_with_scope(name));
	}
	name_scope::~name_scope(){
		OThreadContextSessionImpl::getSession()->namescope_stack_.pop();
	}

	string Initializer::seril(){
		string result;

		result += sformat("type: \"%s\"\n", type.c_str());
		if (value != 0) result += sformat("value: %g\n", value);
		if (minval != 0) result += sformat("min: %g\n", minval);
		if (maxval != 1) result += sformat("max: %g\n", maxval);
		if (meanval != 0) result += sformat("mean: %g\n", meanval);
		if (stdval != 1) result += sformat("std: %g\n", stdval);
		if (sparse != -1) result += sformat("sparse: %d\n", sparse);
		if (variance_norm != VarianceNorm_FAN_IN) result += sformat("value: %g\n", variance_norm_string(variance_norm));
		return result;
	}

	string OTensor::shapestr(){
		string r;
		char buf[100];

		for (int i = 0; i < shape.size(); ++i){
			sprintf(buf, "%d", shape[i]);

			if (i == 0)
				r = buf;
			else
				r = r + "," + buf;
		}
		return r;
	}

	string operator+(const char* s1, const string& s2){
		return string(s1) + s2;
	}

	//
	//    序列化参数
	//
	string OOptimizer::seril(){

		string result;
		if (iter_size != 1) result += sformat("iter_size: %d\n", iter_size);
		if (!test_iter.empty()){
			for (size_t i = 0; i < test_iter.intarraySize(); ++i)
				result += sformat("test_iter: %d\n", test_iter.intval(i));
		}
		if (test_interval != 0) result += sformat("test_interval: %d\n", test_interval);
		if (!test_initialization) result += sformat("test_initialization: %s\n", bool_string(test_initialization));
		if (!base_lr.empty()) result += sformat("base_lr: %g\n", base_lr.floatval());
		if (!display.empty()) result += sformat("display: %d\n", display.intval());
		if (average_loss != 1) result += sformat("average_loss: %d\n", average_loss);
		if (!max_iter.empty()) result += sformat("max_iter: %d\n", max_iter.intval());
		if (!lr_policy.empty()) result += sformat("lr_policy: \"%s\"\n", lr_policy.c_str());
		if (random_seed != -1) result += sformat("random_seed: %d\n", random_seed);

		if (!gamma.empty()) result += sformat("gamma: %g\n", gamma.floatval());
		if (!power.empty()) result += sformat("power: %g\n", power.floatval());
		if (!weight_decay.empty()) result += sformat("weight_decay: %g\n", weight_decay.floatval());
		if (!stepsize.empty()) result += sformat("stepsize: %d\n", stepsize.intval());
		for (size_t i = 0; i < stepvalue.size(); ++i)
			result += sformat("stepvalue: %d\n", stepvalue[i]);

		if (!regularization_type.empty()) result += sformat("regularization_type: \"%s\"\n", regularization_type.c_str());

		if (snapshot != 0) result += sformat("snapshot: %d\n", snapshot);
		if (!snapshot_prefix.empty()) result += sformat("snapshot_prefix: \"%s\"\n", snapshot_prefix.strval().c_str());
		if (snapshot_diff) result += sformat("snapshot_diff: %s\n", bool_string(snapshot_diff));
		result += sformat("solver_mode: %s\n", solver_mode_string(solver_mode));
		//if (!device_ids.empty()) result += sformat("device_id: %d\n", device_ids[0]);
		return result + seril_sub_param();
	}
	
	OLayerOp::OLayerOp(){
		this->layer_id = OThreadContextSession::getSession()->next_layer_id();
	}

	string OLayerOp::scope_name_or_next_auto_name(const string& name){
		string uname = name;
		if (uname.empty()){
			string caffetypename = caffe_type_name();
			map<std::string, int>& layer_last_name_number_map = OThreadContextSessionImpl::getSession()->layers_last_name_number_map_;
			uname = sformat("%s%d", caffetypename.c_str(), ++layer_last_name_number_map[caffetypename.c_str()]);
		}
		return get_name_with_scope(uname);
	}

	string OLayerOp::serial(){

		string param = serial_param();
		string result = "layer{\n";
		result += sformat("name: \"%s\"\n", name.c_str());
		result += sformat("type: \"%s\"\n", caffe_type_name());

		for (int i = 0; i < input.size(); ++i)
			result += sformat("bottom: \"%s\"\n", input[i]->name.c_str());

		for (int i = 0; i < output.size(); ++i)
			result += sformat("top: \"%s\"\n", output[i]->name.c_str());

		if (phase){
			result += sformat(
				"include {\n"
				"phase: %s\n"
				"}\n", phase_string(*phase.get()));
		}

		for (size_t i = 0; i < loss_weight.floatarraySize(); ++i)
			result += sformat("loss_weight: %g\n", loss_weight.floatval(i));

		for (size_t i = 0; i < propagate_down.size(); ++i)
			result += sformat("propagate_down: %d\n", propagate_down[i]);

		if (kernel_mult)
			result += kernel_mult->seril() + "\n";

		if (bias_mult)
			result += bias_mult->seril() + "\n";

		if (!param.empty())
			result = result + param + "\n";

		result += "}";
		return result;
	}

	//
	//    指定要优化的对象，图
	//
	void OOptimizer::minimize(const vector<Tensor>& graphs){
		graph_type = GraphType_FromTensor;
		this->graphs = graphs;
	}

	//
	//    指定要优化的对象，图
	//
	void OOptimizer::minimizeFromPrototxt(const string& graphstr){
		graph_type = GraphType_FromPrototxt;
		this->str_graph = graphstr;
	}

	//
	//    指定要优化的对象，图
	//
	void OOptimizer::minimizeFromFile(const string& graphfile){
		graph_type = GraphType_FromFile;
		this->file_graph = graphfile;
	}

	namespace engine{
		namespace caffe{

			void serial_layer(OLayerOp* layer, vector<OLayerOp*>& layer_order,
				map<OLayerOp*, bool>& layer_state, map<string, OLayerOp*>& output_blob_layer_map){

				if (layer_state[layer])
					return;

				for (int i = 0; i < layer->input.size(); ++i){
					Tensor tensor = layer->input[i];
					string name = tensor->owner->name + "#" + tensor->name;
					OLayerOp* l = output_blob_layer_map[name.c_str()];
					serial_layer(l, layer_order, layer_state, output_blob_layer_map);
				}

				if (!layer_state[layer]){
					layer_state[layer] = true;
					layer_order.push_back(layer);
				}
			};

			//
			//    将计算图编译到caffe支持
			//
			bool buildGraphToFile(const vector<Tensor>& ts, const string& file){
				FILE* f = fopen(file.c_str(), "wb");
				if (!f) return false;

				string pb = buildGraph(ts);
				fwrite(pb.data(), 1, pb.size(), f);
				fclose(f);
				return true;
			}

			//
			//    将计算图编译到caffe支持
			//
			string buildGraph(const vector<Tensor>& ts){

				list<OLayerOp*> stack_;

				//0 is root layer
				for (size_t i = 0; i < ts.size(); ++i)
					stack_.push_back(ts[i]->owner.get());

				//bool is serialed
				map<OLayerOp*, bool> all_layer;
				map<string, OLayerOp*> output_blob_layer_map;
				while (!stack_.empty()){

					OLayerOp* layer = stack_.front();
					stack_.pop_front();

					//对每个输入的blob，记录其owner，然后遍历
					//如果该layer已经处理过，则不再对其input做查找，否则d将在递归模块时造成死循环
					if (all_layer.find(layer) != all_layer.end())
						continue;

					all_layer[layer] = false;
					for (int i = 0; i < layer->output.size(); ++i)
						output_blob_layer_map[(layer->name + "#" + layer->output[i]->name).c_str()] = layer;

					for (int i = 0; i < layer->input.size(); ++i){
						if (layer->input[i]->owner)
							stack_.push_back(layer->input[i]->owner.get());
					}
				}

				vector<OLayerOp*> serial_order;
				for (int i = 0; i < ts.size(); ++i){
					serial_layer(ts[i]->owner.get(), serial_order, all_layer, output_blob_layer_map);
				}

				string net_output;
				int space = 0;
				for (int j = 0; j < serial_order.size(); ++j){
					OLayerOp* l = serial_order[j];
					string layer_string = l->serial();
					char* token = strtok((char*)layer_string.c_str(), "\n");

					while (token){
						if (strchr(token, '}'))
							space--;

						for (int k = 0; k < space; ++k)
							net_output += "    ";

						if (strchr(token, '{'))
							space++;

						net_output += sformat("%s\n", token);
						token = strtok(nullptr, "\n");
					}
				}
				return net_output;
			}


			//
			//    编译一个net，然后可以做inference
			//
			std::shared_ptr<Net> buildNet(const vector<Tensor>& graphs, int phase){
				 
				string net_pb = buildGraph(graphs);
				return loadNetFromPrototxtString(net_pb.c_str(), net_pb.length(), phase);
			}
		}
	};
	
	//
	//    引擎部分
	//
	namespace train{

		namespace caffe{

			//
			//    train回调函数的转发函数
			//
			void trainStepEndCallbackFunc(cc::Solver* solver, int step, float smoothed_loss, void* userData){
				OThreadContextSessionImpl* session = (OThreadContextSessionImpl*)userData;
				if (session->step_end_callback_){
					session->step_end_callback_(session, step, smoothed_loss);
				}
			}

			//
			//    从文件读取数据
			//
			string readfromfile(const string& file){
				FILE* f = fopen(file.c_str(), "rb");
				if (!f){
					printf("read fail: %s\n", file.c_str());
					return "";
				}
				string out;
				int len = 0;
				fseek(f, 0, SEEK_END);
				len = ftell(f);
				fseek(f, 0, SEEK_SET);
				if (len > 0){
					out.resize(len);
					fread((char*)out.data(), 1, len, f);
				}
				fclose(f);
				return out;
			}

			//
			//    训练任务执行
			//
			void run(const Optimizer& optimizer, const std::function<StepEndCallbackFunctional>& stepEndCallback){

				string net_pb;
				switch (optimizer->graph_type){
				case GraphType_FromTensor:
					net_pb = engine::caffe::buildGraph(optimizer->graphs);
					break;

				case GraphType_FromFile:
					net_pb = readfromfile(optimizer->file_graph);
					break;

				case GraphType_FromPrototxt:
					net_pb = optimizer->str_graph;
					break;

				case GraphType_None:
					printf("no set graph_type for optimizer.\n");
					return;
				}

				string solver_pb = optimizer->seril();
				std::shared_ptr<cc::Solver> solver = cc::loadSolverFromPrototxtString(solver_pb.c_str(), net_pb.c_str());

				if (!optimizer->reload_weights.empty())
					solver->net()->weightsFromFile(optimizer->reload_weights.c_str());

				OThreadContextSessionImpl::getSession()->net_pb = net_pb;
				OThreadContextSessionImpl::getSession()->solver_pb = solver_pb;
				OThreadContextSessionImpl::getSession()->solver_ = solver.get();
				OThreadContextSessionImpl::getSession()->step_end_callback_ = stepEndCallback;

				//if we have a valid callback function
				if (stepEndCallback)
					solver->setSetpEndCallback(trainStepEndCallbackFunc, OThreadContextSession::getSession());
				
				if (!optimizer->device_ids.empty()){
					solver->solve(optimizer->device_ids.size(), optimizer->device_ids.data());
				}
				else{
					solver->solve();
				}

				OThreadContextSessionImpl::getSession()->solver_ = nullptr;
			}
		};
	};

	namespace layers{

		string OSplit::serial_param(){
			string part;
			if (axis != 1) part = sformat("axis: %d\n", axis);
			for (int i = 0; i < slice_point.size(); ++i)
				part += sformat("slice_point: %d\n", slice_point[i]);

			if (part.empty()) return "";
			return "slice_param {\n" + part + "}";
		}

		string OConcat::serial_param(){
			string part;
			if (axis != 1) part = sformat("axis: %d\n", axis);
			if (part.empty()) return "";
			return "concat_param {\n" + part + "}";
		}

		string OTranspose::serial_param(){
			string part;
			for (int i = 0; i < order.size(); ++i)
				part += sformat("order: %d\n", order[i]);

			if (order.empty()) return "";
			return "permute_param {\n" + part + "}";
		}

		string OReshape::serial_param(){
			string part = "";
			if (axis != 0) part += sformat("axis: %d\n", axis);

			if (!new_dims.empty()){
				part += "shape {\n";
				for (int i = 0; i < new_dims.size(); ++i)
					part += sformat("dim: %d\n", new_dims[i]);
				part += "}\n";
			}
			return "reshape_param {\n" + part + "}";
		}

		string OAdd::serial_param(){
			string part = "operation: SUM\n";
			if (!stable_prod_grad) part += sformat("stable_prod_grad: %s\n", bool_string(stable_prod_grad));

			for (int i = 0; i < coeff.size(); ++i)
				part += sformat("coeff: %g\n", coeff[i]);
			return "eltwise_param {\n" + part + "}";
		}

		string OProduct::serial_param(){
			string part = "operation: PROD\n";
			if (!stable_prod_grad) part += sformat("stable_prod_grad: %s\n", bool_string(stable_prod_grad));
			return "eltwise_param {\n" + part + "}";
		}

		string OSoftmax::serial_param(){
			string part;
			if (axis != 1) part += sformat("axis: %d\n", axis);
			if (hard_ratio != 1.0f) part += sformat("hard_ratio: %g\n", hard_ratio);
			if (!hard_mining_label.empty()) part += sformat("hard_mining_label: %d\n", hard_mining_label.intval());
			if (!cutting_point.empty()) part += sformat("cutting_point: %g\n", cutting_point.floatval());
			if (normalize_type != "Softmax") part += sformat("normalize_type: %g\n", normalize_type.c_str());
			for (size_t i = 0; i < class_weight.floatarraySize(); ++i)
				part += sformat("class_weight: %g\n", class_weight.floatval(i));
			if (part.empty()) return "";
			return "softmax_param {\n" + part + "\n}";
		}

		string OMax::serial_param(){
			string part = "operation: MAX\n";
			return "eltwise_param {\n" + part + "}";
		}

		string ODropout::serial_param(){
			string part = sformat("dropout_ratio: %g\n", 1 - keep_prob);
			return "dropout_param {\n" + part + "}";
		}

		string OScale::serial_param(){

			string part;
			part = sformat("bias_term: %s\n", bool_string(bias_term));
			if (axis != 1) part += sformat("axis: %d\n", axis);
			if (num_axes != 1) part += sformat("num_axes: %d\n", num_axes);
			if (bias_filler) part += bias_filler->seril();
			if (part.empty()) return "";

			return "scale_param {\n" + part + "}";
		}

		string OBatchNorm::serial_param(){

			string part;
			//part += sformat("use_global_stats: %s\n", bool_string(true));
			if (moving_average_fraction != 0.999f) part += sformat("moving_average_fraction: %g\n", moving_average_fraction);
			if (eps != 1e-5f) part += sformat("eps: %g\n", eps);
			if (part.empty()) return "";

			return "batch_norm_param {\n" + part + "}";
		}

		string OPooling2D::serial_param(){

			string result = sformat(
				"pooling_param {\n"
				"pool: %s\n",
				pool_method_string(method));

			if (global_pooling){
				result += sformat("global_pooling: %s\n", bool_string(global_pooling));
			}
			else{
				//卷积核的定义
				if (kernel[0] != kernel[1]){
					result += sformat(
						"kernel_h: %d\n"
						"kernel_w: %d\n"
						, kernel[0], kernel[1]);
				}
				else{
					result += sformat("kernel_size: %d\n", kernel[0]);
				}
			}

			if (padding_size[0] != padding_size[1]){
				result += sformat(
					"pad_w: %d\n"
					"pad_h: %d\n"
					, padding_size[0], padding_size[1]);
			}
			else{
				if (padding_size[0] != 0)
					result += sformat("pad: %d\n", padding_size[0]);
			}

			if (strides[0] != strides[1]){
				result += sformat(
					"stride: %d\n"
					"stride: %d\n"
					, strides[0], strides[1]);
			}
			else{
				if (strides[0] != 1)
					result += sformat("stride: %d\n", strides[0]);
			}
			result += "}";
			return result;
		}

		string ODense::serial_param(){
			string part = sformat("num_output: %d\n", units);
			if (!bias_term) part += sformat("bias_term: %s\n", bool_string(bias_term));

			if (weight_initializer)
				part = part + "weight_filler {\n" + weight_initializer->seril() + "\n}\n";

			if (bias_initializer)
				part = part + "bias_filler {\n" + bias_initializer->seril() + "\n}\n";

			if (axis != -1) part += sformat("axis: %d\n", axis);
			if (transpose) part += sformat("transpose: %s\n", bool_string(transpose));
			return "inner_product_param {\n" + part + "}";
		}

		string OROIPooling::serial_param(){
			string part;
			part = part + sformat("pooled_w: %d\n", pooled_w);
			part = part + sformat("pooled_h: %d\n", pooled_h);
			part = part + sformat("spatial_scale: %f\n", spatial_scale);
			return "roi_pooling_param {\n" + part + "}";
		}

		string OConv2D::serial_param(){

			string result = sformat(
				"convolution_param {\n"
				"num_output: %d\n",
				kernel[2]);
			if (!bias_term) result += sformat("bias_term: %s\n", bool_string(bias_term));

			//卷积核的定义
			if (kernel[0] != kernel[1]){
				result += sformat(
					"kernel_size: %d\n"
					"kernel_size: %d\n"
					, kernel[0], kernel[1]);
			}
			else{
				result += sformat("kernel_size: %d\n", kernel[0]);
			}

			if (padding_size[0] != padding_size[1]){
				result += sformat(
					"pad: %d\n"
					"pad: %d\n"
					, padding_size[0], padding_size[1]);
			}
			else{
				if (padding_size[0] != 0)
					result += sformat("pad: %d\n", padding_size[0]);
			}

			if (strides[0] != strides[1]){
				result += sformat(
					"stride: %d\n"
					"stride: %d\n"
					, strides[0], strides[1]);
			}
			else{
				if (strides[0] != 1)
					result += sformat("stride: %d\n", strides[0]);
			}

			if (dilations[0] != dilations[1]){
				result += sformat(
					"dilation: %d\n"
					"dilation: %d\n"
					, dilations[0], dilations[1]);
			}
			else{
				if (dilations[0] != 1)
					result += sformat("dilation: %d\n", dilations[0]);
			}

			if (kernel_initializer){
				result += sformat("weight_filler {\n%s}\n",
					kernel_initializer->seril().c_str());
			}

			if (bias_initializer){
				result += sformat("bias_filler {\n%s}\n",
					bias_initializer->seril().c_str());
			}

			result += "}";
			return result;
		}

		///////////////////////////////////////////////////////////////////////////////

		//
		//    数据输入层的定义
		//
		string OInput::serial_param(){
			string part;
			for (int i = 0; i < dims.size(); ++i)
				part += sformat("dim: %d\n", dims[i]);
			return "input_param {\nshape {\n" + part + "}\n}";
		}

		Tensor input(const vector<int>& dims, const string& name){
			OInput* pinput = new OInput();
			pinput->dims = dims;

			LayerOp layer(pinput);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = dims;
			layer->output[0] = blob;
			return layer->output[0];
		}

		//
		//    数据层的定义
		//

		string OCustom::serial_param(){
			string result = "cpp_param {\n";
			if (!cpp_param_str.empty()) result += sformat("param_str: %s\n", cpp_param_str.c_str());
			result += sformat("type: \"%s\"\n", cpp_type.c_str());
			result += "}";
			return result;
		}

		vector<Tensor> data(const string& cpp_type, const vector<string>& output, const string& name, const string& param_str){

			OCustom* pdata = new OCustom();
			pdata->cpp_type = cpp_type;
			pdata->cpp_param_str = param_str;

			LayerOp layer(pdata);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(output.size());

			for (int i = 0; i < output.size(); ++i){

				Tensor blob(new OTensor());
				blob->name = output[i];
				blob->owner = layer;
				layer->output[i] = blob;
			}
			return layer->output;
		}

		//
		//    自定义层1
		//
		vector<Tensor> custom(const string& cpp_type, const vector<Tensor>& input, const vector<string>& output, const string& name, const string& param_str){

			OCustom* pdata = new OCustom();
			pdata->cpp_type = cpp_type;
			pdata->cpp_param_str = param_str;

			LayerOp layer(pdata);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input = input;
			layer->output.resize(output.size());

			for (int i = 0; i < output.size(); ++i){

				Tensor blob(new OTensor());
				blob->name = layer->name + "/" + output[i];
				blob->owner = layer;
				blob->shape = { 1, 1, 1, 1 };
				layer->output[i] = blob;
			}
			return layer->output;
		}

		//
		//    自定义层2
		//
		Tensor custom(const string& cpp_type, const vector<Tensor>& input, const string& output, const string& name, const string& param_str){

			OCustom* pdata = new OCustom();
			pdata->cpp_type = cpp_type;
			pdata->cpp_param_str = param_str;

			LayerOp layer(pdata);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input = input;
			layer->output.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name + "/" + output;
			blob->owner = layer;
			blob->shape = { 1, 1, 1, 1 };
			layer->output[0] = blob;
			return blob;
		}

		//
		//    自定义层3
		//
		Tensor custom(const string& cpp_type, const Tensor& input, const vector<int>& output_shape, const string& name, const string& param_str){

			OCustom* pdata = new OCustom();
			pdata->cpp_type = cpp_type;
			pdata->cpp_param_str = param_str;

			LayerOp layer(pdata);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input = { input };
			layer->output.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;

			if (output_shape.empty())
				blob->shape = input->shape;
			else
				blob->shape = output_shape;
			layer->output[0] = blob;
			return blob;
		}

		//
		//    卷积层的定义
		//    x:        tensor
		//              需要卷积的tensor
		//
		//    kernel:   3-d array
		//              卷积核的大小，这里是2维，指定为height, width, output
		//
		//    padding:    "valid"or "same"
		//              指定padding的实现方式，valid即卷积后尺寸，无padding，same即卷积后尺寸和x一致
		//
		//    strides:  2-d array, height, width
		//              指定步长
		//
		//    dilations: 2-d array, height, width
		//              卷积的膨胀尺寸
		//
		//    name:     指定卷积层名称
		//              默认是为空，即自动生成的名称
		//
		Tensor conv2d(const Tensor&  x, const vector<int>& kernel, const string& padding,
			const vector<int>& strides, const vector<int>& dilations, const string& name){

			OConv2D* conv = new OConv2D();
			conv->kernel = kernel;
			conv->padding = padding;
			conv->strides = strides;
			conv->padding_size.resize(2);
			conv->dilations = dilations;

#if 0
			//不能够内部分配，否则出错
			//我们一般默认卷积的权重初始化方式会是xavier
			conv->kernel_initializer.reset(new Initializer());
			conv->bias_initializer.reset(new Initializer());

			conv->kernel_initializer->type = "gaussian";
			conv->kernel_initializer->stdval = 0.01;
			conv->bias_initializer->type = "constant";
			conv->bias_initializer->value = 0;
#endif

			LayerOp layer(conv);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

#if 0
			//不能够内部分配，否则出错
			if (hasmult){
				layer->kernel_mult.reset(new ParamSpecMult());
				layer->bias_mult.reset(new ParamSpecMult());

				layer->kernel_mult->decay_mult = 0;
				layer->kernel_mult->lr_mult = 1;

				layer->bias_mult->decay_mult = 0;
				layer->bias_mult->lr_mult = 2;
			}
#endif

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;

			//shape:  n, c, h, w
			//kernel: h, w, output
			blob->shape[0] = x->shape[0];
			blob->shape[1] = kernel[2];

			if (padding == "valid"){
				conv->padding_size[0] = 0;
				conv->padding_size[1] = 0;
			}
			else if (padding == "same"){
				conv->padding_size[0] = (dilations[0] * (kernel[0] - 1) + 1) / 2;
				conv->padding_size[1] = (dilations[1] * (kernel[1] - 1) + 1) / 2;
			}

			blob->shape[2] = compute_std_dnn_output_shape(x->shape[2], kernel[0], strides[0], conv->padding_size[0], dilations[0]);
			blob->shape[3] = compute_std_dnn_output_shape(x->shape[3], kernel[1], strides[1], conv->padding_size[1], dilations[1]);

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor transpose(const Tensor&  x, vector<int> order, const string& name, bool inplace){

			OTranspose* r = new OTranspose();
			r->order = order;

			LayerOp layer(r);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			blob->shape = x->shape;

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor reshape(const Tensor&  x, vector<int> new_dims, const string& name){

			OReshape* r = new OReshape();
			r->new_dims = new_dims;

			LayerOp layer(r);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = x->shape;

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor concat(const vector<Tensor>& tensors, int axis, const string& name){

			OConcat* c = new OConcat();
			c->axis = axis;

			LayerOp layer(c);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(tensors.size());
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = tensors[0]->shape;

			layer->input = tensors;
			layer->output[0] = blob;
			return blob;
		}

		Tensor max_pooling2d(const Tensor&  x, const vector<int>& kernel, const vector<int>& strides, const vector<int>& padding, bool global_pooling, const string& name){
			OPooling2D* pool = new OPooling2D();
			pool->kernel = kernel;
			pool->strides = strides;
			pool->method = PoolMethod_MAX;
			pool->global_pooling = global_pooling;
			pool->padding_size = padding;

			LayerOp layer(pool);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = x->shape;

			if (!global_pooling){
				//shape:  n, c, h, w
				//kernel: h, w, output
				//这里的shape计算是错误的
				blob->shape[2] = compute_std_dnn_output_shape(x->shape[2], kernel[0], strides[0], pool->padding_size[0]);
				blob->shape[3] = compute_std_dnn_output_shape(x->shape[3], kernel[1], strides[1], pool->padding_size[1]);
			}
			else{
				blob->shape[2] = 1;
				blob->shape[3] = 1;
			}

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor avg_pooling2d(const Tensor&  x, const vector<int>& kernel, const vector<int>& strides, const vector<int>& padding, bool global_pooling, const string& name){

			OPooling2D* pool = new OPooling2D();
			pool->kernel = kernel;
			pool->strides = strides;
			pool->method = PoolMethod_AVE;
			pool->global_pooling = global_pooling;
			pool->padding_size = padding;

			LayerOp layer(pool);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = x->shape;

			if (!global_pooling){
				//shape:  n, c, h, w
				//kernel: h, w, output
				blob->shape[2] = compute_std_dnn_output_shape(x->shape[2], kernel[0], strides[0], pool->padding_size[0]);
				blob->shape[3] = compute_std_dnn_output_shape(x->shape[3], kernel[1], strides[1], pool->padding_size[1]);
			}
			else{
				blob->shape[2] = 1;
				blob->shape[3] = 1;
			}

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor dense(const Tensor&  x, int units, const string& name, bool bias_term){

			ODense* d = new ODense();
			d->units = units;
			d->bias_term = bias_term;

#if 0
			//不能内部构造，否则外边没法修改
			//我们一般默认卷积的权重初始化方式会是xavier
			d->weight_initializer.reset(new Initializer());
			d->bias_initializer.reset(new Initializer());

			d->weight_initializer->type = "gaussian";
			d->bias_initializer->type = "constant";
#endif

			LayerOp layer(d);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

#if 0
			layer->kernel_mult.reset(new ParamSpecMult());
			layer->bias_mult.reset(new ParamSpecMult());

			layer->kernel_mult->decay_mult = 0;
			layer->kernel_mult->lr_mult = 2;

			layer->bias_mult->decay_mult = 0;
			layer->bias_mult->lr_mult = 1;
#endif

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = x->shape;
			blob->shape[1] = units;
			blob->shape[2] = 1;
			blob->shape[3] = 1;

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor add(const Tensor&  a, const Tensor&  b, const string& name){
			LayerOp layer(new OAdd());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(2);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = a->shape;
			layer->input[0] = a;
			layer->input[1] = b;
			layer->output[0] = blob;
			return blob;
		}

		Tensor maxop(const Tensor&  a, const Tensor&  b, const string& name){
			LayerOp layer(new OMax());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(2);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = a->shape;
			layer->input[0] = a;
			layer->input[1] = b;
			layer->output[0] = blob;
			return blob;
		}

		Tensor softmax(const Tensor&  x, const string& name, bool inplace){
			LayerOp layer(new OSoftmax());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			blob->shape = x->shape;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor relu(const Tensor&  x, const string& name, bool inplace){
			LayerOp layer(new OReLU());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			blob->shape = x->shape;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor batch_norm_only(const Tensor&  x, const string& name, bool inplace){
			OBatchNorm* bn = new OBatchNorm();

			LayerOp layer(new OBatchNorm());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			blob->shape = x->shape;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor roi_pooling(const Tensor&  feature_map, const Tensor&  rois, int pooled_w, int pooled_h, float spatial_scale, const string& name){
			OROIPooling* pool = new OROIPooling();
			pool->pooled_w = pooled_w;
			pool->pooled_h = pooled_h;
			pool->spatial_scale = spatial_scale;
			pool->name = pool->scope_name_or_next_auto_name(name);

			LayerOp layer(pool);
			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = { 1, 1, 1, 1 };
			layer->input = { feature_map, rois };
			layer->output = { blob };
			return blob;
		}

		Tensor scale(const Tensor&  x, bool bias_term, const string& name, bool inplace){
			OScale* scale = new OScale();
			scale->bias_term = bias_term;

			LayerOp layer(scale);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			blob->shape = x->shape;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor batch_norm(const Tensor&  x, bool bias_term, const string& name, bool inplace){
			Tensor o;
			o = batch_norm_only(x, name.empty() ? "" : name + "/bn", inplace);
			o = scale(o, bias_term, name.empty() ? "" : name + "/scale", inplace);
			return o;
		}

		Tensor dropout(const Tensor&  x, float keep_prob, const string& name, bool inplace){

			ODropout* drop = new ODropout();
			drop->keep_prob = keep_prob;

			LayerOp layer(drop);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			blob->shape = x->shape;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}
	}

	// Return the current learning rate. The currently implemented learning rate
	// policies are as follows:
	//    - fixed: always return base_lr.
	//    - step: return base_lr * gamma ^ (floor(iter / step))
	//    - exp: return base_lr * gamma ^ iter
	//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
	//    - multistep: similar to step but it allows non uniform steps defined by
	//      stepvalue
	//    - poly: the effective learning rate follows a polynomial decay, to be
	//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
	//    - sigmoid: the effective learning rate follows a sigmod decay
	//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
	//
	// where base_lr, max_iter, gamma, step, stepvalue and power are defined
	// in the solver parameter protocol buffer, and iter is the current iteration.
	namespace learningrate{

		//
		//    fixed学习率策略  always return base_lr.
		//
		LearningRatePolicy fixed(float base_lr){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "fixed";
			return lrp;
		}

		//
		//    step学习率策略  return base_lr * gamma ^ (floor(iter / step_size))
		//
		LearningRatePolicy step(float base_lr, float gamma, int step_size){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "step";
			lrp->gamma = gamma;
			lrp->stepsize = step_size;
			return lrp;
		}

		//
		//    exp学习率策略  return base_lr * gamma ^ iter
		//
		LearningRatePolicy exp(float base_lr, float gamma){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "exp";
			lrp->gamma = gamma;
			return lrp;
		}

		//
		//    inv学习率策略  return base_lr * (1 + gamma * iter) ^ (- power)
		//
		LearningRatePolicy inv(float base_lr, float gamma, float power){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "inv";
			lrp->gamma = gamma;
			lrp->power = power;
			return lrp;
		}

		//
		//    multistep学习率策略  similar to step but it allows non uniform steps defined by
		//      stepvalue
		//
		LearningRatePolicy multistep(float base_lr, float gamma, const vector<int>& stepvalue){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "multistep";
			lrp->gamma = gamma;
			lrp->stepvalue = stepvalue;
			return lrp;
		}

		//
		//    poly学习率策略  the effective learning rate follows a polynomial decay, to be
		//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
		//
		LearningRatePolicy poly(float base_lr, float power){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "poly";
			lrp->power = power;
			return lrp;
		}

		//
		//    sigmoid学习率策略  the effective learning rate follows a sigmod decay
		//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
		//
		LearningRatePolicy sigmoid(float base_lr, float gamma, int stepsize){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "sigmoid";
			lrp->gamma = gamma;
			lrp->stepsize = stepsize;
			return lrp;
		}
	};

	namespace optimizer{

		string AdaptiveMomentEstimation::seril_sub_param(){
			string result;
			result += sformat("momentum: %g\n", beta1);
			result += sformat("momentum2: %g\n", beta2);
			result += sformat("delta: %g\n", delta);
			return result + "solver_type: ADAM";
		}

		Optimizer stochasticGradientDescent(LearningRatePolicy lr){
			StochasticGradientDescent* sgd = new StochasticGradientDescent();
			sgd->setlr(lr);

			Optimizer op(sgd);
			return op;
		}

		Optimizer momentumStochasticGradientDescent(LearningRatePolicy lr, float momentum){
			StochasticGradientDescent* sgd = new StochasticGradientDescent();
			sgd->setlr(lr);
			sgd->momentum = momentum;

			Optimizer op(sgd);
			return op;
		}

		Optimizer adaptiveMomentEstimation(LearningRatePolicy lr, float beta1, float beta2, float delta){
			AdaptiveMomentEstimation* adam = new AdaptiveMomentEstimation();
			adam->setlr(lr);
			adam->beta1 = beta1;
			adam->beta2 = beta2;
			adam->delta = delta;

			Optimizer op(adam);
			return op;
		}
	};

	//
	//     loss的定义
	//
	namespace loss{

		string OSoftmaxCrossEntropy::serial_param(){
			string part;
			if (axis != 1) part += sformat("axis: %d\n", axis);
			if (hard_ratio != 1.0f) part += sformat("hard_ratio: %g\n", hard_ratio);
			if (!hard_mining_label.empty()) part += sformat("hard_mining_label: %d\n", hard_mining_label.intval());
			if (!cutting_point.empty()) part += sformat("cutting_point: %g\n", cutting_point.floatval());
			if (normalize_type != "Softmax") part += sformat("normalize_type: %s\n", normalize_type.c_str());
			for (size_t i = 0; i < class_weight.floatarraySize(); ++i)
				part += sformat("class_weight: %g\n", class_weight.floatval(i));

			string softmax_param = part.empty() ? "" : "softmax_param {\n" + part + "\n}";
			string loss_param;
			if (!ignore_label.empty()) loss_param += sformat("ignore_label: %d\n", ignore_label.intval());
			if (normalize) loss_param += sformat("normalize: %s\n", bool_string(normalize));
			if (!loss_param.empty()) softmax_param += sformat("\nloss_param{\n%s}", loss_param.c_str());
			return softmax_param;
		}

		string OSmoothL1::serial_param(){
			if (sigma.empty())
				return "";
			return sformat("smooth_l1_loss_param {\nsigma: %g\n}", sigma.floatval());
		}

		//
		//    具体交叉熵损失的函数定义
		//
		Tensor softmax_cross_entropy(const Tensor&  x, const Tensor&  y, const string& name, Tensor* loss_weight, bool normalize, DynamicValue ignore_label){

			OSoftmaxCrossEntropy* loss_ = new OSoftmaxCrossEntropy();
			loss_->ignore_label = ignore_label;
			loss_->normalize = normalize;

			LayerOp layer(loss_);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input.push_back(x);
			layer->input.push_back(y);
			if (loss_weight)
				layer->input.push_back(*loss_weight);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = { 1 };
			layer->output.push_back(blob);
			return blob;
		}

		//
		//    具体交叉熵损失的函数定义
		//
		Tensor smooth_l1(const Tensor&  x, const Tensor&  y, float sigma, const string& name, const vector<Tensor>& loss_weights){

			OSmoothL1* loss_ = new OSmoothL1();
			loss_->sigma = sigma;

			LayerOp layer(loss_);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input.push_back(x);
			layer->input.push_back(y);
			for (int i = 0; i < loss_weights.size(); ++i)
				layer->input.push_back(loss_weights[i]);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = { 1 };
			layer->output.push_back(blob);
			return blob;
		}
	};

	//
	//    关于评测方法
	//
	namespace metric{

		string OClassifyAccuracy::serial_param(){
			string part;
			if (top_k != 1) part += sformat("top_k: %d\n", top_k);
			if (axis != 1) part += sformat("axis: %d\n", axis);
			if (!ignore_label.empty()) part += sformat("ignore_label: %d\n", ignore_label.intval());
			if (part.empty()) return "";
			return sformat("accuracy_param {\n%s\n}", part.c_str());
		}

		//
		//    accuracy
		//
		Tensor classifyAccuracy(const Tensor&  x, const Tensor&  y, const string& name){

			OClassifyAccuracy* acc = new OClassifyAccuracy();
			LayerOp layer(acc);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input.resize(2);
			layer->output.resize(1);
			layer->phase.reset(new Phase(Phase_TEST));

			layer->input[0] = x;
			layer->input[1] = y;
			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			blob->shape = { 1 };
			layer->output[0] = blob;
			return blob;
		}
	}
}