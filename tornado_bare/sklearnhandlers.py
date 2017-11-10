#!/usr/bin/python
from pymongo import MongoClient
import tornado.web
from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from basehandler import BaseHandler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pickle
from bson.binary import Binary
import json
import numpy as np

from sklearn import cross_validation

class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        fvals = [float(val) for val in vals]
        label = data['label']
        sess  = data['dsid']
        print(label)
        #Updated to save properly
        dbid = self.db.labeledinstances.insert(
                {"feature":fvals,"label":label,"dsid":sess}
            )
        self.write_json({"id":str(dbid),"feature":fvals,"label":label})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

# updated to be a dictionary
class UpdateModelForDatasetId(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        dsid = self.get_int_arg("dsid",default=0)
        numNeighbors = self.get_int_arg("numNeighbors",default=3)
        print(numNeighbors)

        # create feature vectors from database
        f=[]
        for a in self.db.labeledinstances.find({"dsid":dsid}):
            f.append([float(val) for val in a['feature']])

        # create label vector from database
        l=[]
        for a in self.db.labeledinstances.find({"dsid": dsid}):
            l.append(a['label'])

        # fit the model to the data
        self.classifiers = [
            KNeighborsClassifier(n_neighbors = numNeighbors),
            svm.SVC(kernel='linear', C=100),
            LogisticRegression(),
            MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        ]

        self.classifiers_pkl = []

        self.classifiers_accuracy = []

        acc = -1
        if l:
            # c1.fit(f, l)
            # training
            for classifier in self.classifiers:
                classifier.fit(f, l)
                self.classifiers_pkl.append(pickle.dumps(classifier))
                #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
                X_train, X_test, y_train, y_test = cross_validation.train_test_split(f, l, test_size=0.2, random_state=0)
                scores = cross_validation.cross_val_score(classifier, f, l, cv=10)
                self.classifiers_accuracy.append(scores.mean())

            print(self.classifiers_accuracy)
            # lstar = c1.predict(f)
            # self.clf = classifiers
            # acc = sum(lstar == l)/float(len(l))
            # bytes = pickle.dumps(c1)
            self.db.models.update(
                {
                    "dsid": dsid
                },
                {
                    "$set": {
                        "model_knn": Binary(self.classifiers_pkl[0]),
                        "model_svm": Binary(self.classifiers_pkl[1]),
                        "model_lr": Binary(self.classifiers_pkl[2]),
                        "model_mlp": Binary(self.classifiers_pkl[3]),
                    }
                },
                upsert=True
            )
            print("Retrained the models")

            self.write_json({
                "model_knn": self.classifiers_accuracy[0],
                "model_svm": self.classifiers_accuracy[1],
                "model_lr": self.classifiers_accuracy[2],
                "model_mlp": self.classifiers_accuracy[3]
            })

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        # self.write_json({"resubAccuracy": acc})

class PredictOneFromDatasetId(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        dsid = data['dsid']
        model = data['model']

        fvals = [float(val) for val in vals]
        fvals = np.array(fvals).reshape(1, -1)

        #if self.clf == []:
        print('Loading Model From DB')
        tmp = self.db.models.find_one({"dsid": dsid})
        if tmp:
            self.clf = pickle.loads(tmp[model])
            print(model)
        else:
            print("Shit! We got to the else")
            # c1 = KNeighborsClassifier(n_neighbors=3)
            # self.clf = c1
            # bytes = pickle.dumps(c1)
            # self.db.models.update(
            #     {
            #         "dsid": dsid
            #     },
            #     {
            #         "$set": {
            #             "model": Binary(bytes)
            #         }
            #     },
            #     upsert=True
            # )
        predLabel = self.clf.predict(fvals)
        print(predLabel)
        self.write_json({"prediction": str(predLabel)})
