from openml.apiconnector import APIConnector
from sklearn import tree
import subprocess

apikey = '6bf2b9754012ef63cb8473a41e8e37bb'

def load(dataset_id):
    print 'Loadding data_id %d' % (dataset_id)
    connector = APIConnector(apikey=apikey)
    dataset = connector.download_dataset(dataset_id)
    return dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)

def export_tree( clf, features, classes, filename="tree.png" ):

    tree.export_graphviz(
        clf,
        feature_names = features,
        class_names   = classes,
        out_file      = 'tree.dot'
    )

    command = ["dot", "-Tpng", "tree.dot", "-o", filename ]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")
