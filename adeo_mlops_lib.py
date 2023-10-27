import cloudpickle as cp 
import datetime 

class mlopslib():

    
    lib_version = '0.0.0.1' # draft



    # ------------------------------

    #internal objects 
    _metadata = {
                    'model_name': '',
                    'author': '',
                    'description':'',
                    'other_information': [],
                    'latest_version': '',
                    'other_available_versions': []
                }

    _train_event = {
         'event_ts'  : 'get the timestamp',
         'dataset'   : 'link to the correct dataset',
         'eval': 'link to the evaluation metric '
        }

    _model = None
    _dataset = None
    _tf = None
    _of = None
    _ff = None
    _if = None

# preparation inernal functions 

    def _save(self):
        with open(self.filename, 'wb') as f:
            cp.dump(self, f)

    def set_transform(self, fn):
        self._tf = fn

    def set_output(self, fn):
        self._of = fn
      
    def set_train(self, fn):
        self._ff = fn

    def set_inference(self, fn):
        self._if = fn

    def _eval(self, eval):
        return metric_object.accuracy_score(eval.dataset, eval.labelset)

    def __init__(self, model, dataset, transformfn, outputfn, trainningfn, inferencefn):
        self._model = model
        self._dataset = dataset
        self._tf = transformfn
        self._of = outputfn
        self._ff = trainningfn
        self._if = inferencefn

# functionalities provided by the lib
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return cp.load(f)

    def deploy(self, serialized_model): 

        version = str(datetime.datetime.now().timestamp())
        serialized_model = serialized_model.replace('<version>',version)

        self.filename = serialized_model
        self._save()
        print(serialized_model)

    def metadata(self):
        return _metadata

    def train(self):
        self._ff(self._dataset['train_X'], self._dataset['train_y'])
        return 'some training event metadata'

    def predict(self, data):
        return self._if(data) 

    def transform(self, input):
        return self._tf(input)
  
    def output(self, job_id, input, transformed, inference):
        return self._of(job_id, input, transformed, inference)


# ------------------Not implemented yet

    def get_model_versions():
        raise NotImplementedError()

    def load_version():
        raise NotImplementedError()

#self._eval(self.eval)
#a_train_event = _train_event_template
#a_train_event['eval'] = self._eval(eval)
#_metadata['trainings'].append(_train_event)