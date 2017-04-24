import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import datetime
import os

high_level_features = [
 'DER_mass_MMC',
 'DER_mass_transverse_met_lep',
 'DER_mass_vis',
 'DER_pt_h',
 'DER_deltar_tau_lep',
 'DER_pt_tot',
 'DER_sum_pt',
 'DER_pt_ratio_lep_tau',
 'DER_met_phi_centrality',
 'DER_lep_eta_centrality']

low_level_features = [
 'PRI_tau_pt',
 'PRI_tau_eta',
 'PRI_tau_phi',
 'PRI_lep_pt',
 'PRI_lep_eta',
 'PRI_lep_phi',
 'PRI_met',
 'PRI_met_phi',
 'PRI_met_sumet',
 'PRI_jet_num',
 'PRI_jet_leading_pt',
 'PRI_jet_all_pt']


def mlp(x, weights, biases, n_layers, p_dropout=1.0):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, p_dropout)
    previous_layer = layer_1
    for i in range(1, n_layers):
        layer_label = 'h' + str(i + 1)
        bias_label = 'b' + str(i + 1)
        layer = tf.add(tf.matmul(previous_layer, weights[layer_label]), biases[bias_label])
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, p_dropout)
        previous_layer = layer
    out_layer_label = 'h' + str(n_layers + 1)
    out_bias_label = 'b' + str(n_layers + 1)
    out_layer = tf.matmul(previous_layer, weights[out_layer_label]) + biases[out_bias_label]
    return out_layer     

def get_weights(n_input,n_output,n_units_per_h_layer):

    units_per_layer = [n_input] + n_units_per_h_layer + [n_output]
    weights = dict()
    for i in range(0,len(units_per_layer) - 1):
        current_layer = 'h' + str(i + 1)
        weights.update({current_layer:tf.Variable(tf.random_uniform([units_per_layer[i], units_per_layer[i+1]],minval=-1,maxval=1))})
    return weights

def get_regularizer(weights):
    weights_list = weights.values()
    losses = [tf.nn.l2_loss(w) for w in weights_list]
    total_loss = tf.add_n(losses)
    return total_loss    
    
def get_biases(n_input,n_output,n_units_per_h_layer):

    units_per_layer = [n_input] + n_units_per_h_layer + [n_output]
    biases = dict()
    for i in range(0,len(units_per_layer) - 1):
        current_layer = 'b' + str(i + 1)
        biases.update({current_layer:tf.Variable(tf.random_uniform([units_per_layer[i + 1]],minval=-1,maxval=1))})
    return biases

def test_model(params,features,chkpt_file=None):
    print params
    data_path = "../higgs_data/atlas-higgs-challenge-2014-v2.csv"

    data = pd.read_csv(data_path)
    data['DER_mass_MMC'] = data['DER_mass_MMC'].replace(-999,data['DER_mass_MMC'].median())
    
    testing_data = data[720000:]

    del data

    test_inputs = testing_data[features]

    test_inputs = (test_inputs - test_inputs.mean())/test_inputs.std()
    test_labels = pd.get_dummies(testing_data['Label'])

    # Parameters
    n_hidden_layers = int( params['n_hidden_layers'])
    units_per_layer = int( params['units_per_layer'])
   
    # Network Parameters
    n_input = len(features) # data input
    n_classes = 2 # total classes (signal, background)

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    p_dropout = tf.placeholder("float", None)

    # Construct model
    weights = get_weights(n_input,n_classes,[units_per_layer]*n_hidden_layers)
    biases = get_biases(n_input,n_classes,[units_per_layer]*n_hidden_layers)
    vars_dict = weights.copy().update(biases)

    #pred = multilayer_perceptron(x, weights, biases)
    pred = mlp(x,weights,biases,n_hidden_layers,p_dropout=p_dropout)
    # Define loss and optimizer

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    # Initializing the variables
    saver = tf.train.Saver(vars_dict)

    # Launch the graph
    with tf.Session() as sess:
        saver.restore(sess, "/tmp/model.ckpt")
        print weights['h1'].eval()
        # Training cycle
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        test_cost = sess.run(cost, feed_dict={x: test_inputs,
                                                  y: test_labels,
                                                  p_dropout: 1.0})
        print("Test Set Accuracy : ", accuracy.eval({x: test_inputs, y: test_labels,p_dropout:1.0}))
        
#params = dict(learning_rate=np.array([.0001]), n_hidden_layers=np.array([1]), units_per_layer=np.array([50]),beta=np.array([1]))
#test_model(params, high_level_features + low_level_features)

def train_model(params, chkpt_file_name):
    print params
    data_path = "../higgs_data/atlas-higgs-challenge-2014-v2.csv"

    data = pd.read_csv(data_path)
    data['DER_mass_MMC'] = data['DER_mass_MMC'].replace(-999,data['DER_mass_MMC'].median())
    
    training_data = data[120000:720000]
    training_data.index = range(600000)
    testing_data = data[720000:]
    training_set_size = 100000

    del data
    features = params['features']
    train_inputs = training_data[features]
    test_inputs = testing_data[features]

    train_inputs = (train_inputs - train_inputs.mean())/train_inputs.std()
    test_inputs = (test_inputs - test_inputs.mean())/test_inputs.std()
    train_labels = pd.get_dummies(training_data['Label'])
    test_labels = pd.get_dummies(testing_data['Label'])

    # Parameters
    learning_rate =float( params['learning_rate'])
    n_hidden_layers = int( params['n_hidden_layers'])
    units_per_layer = int( params['units_per_layer'])
    beta = int( params['beta'])
    
    now = str(datetime.datetime.now()).replace(' ','-').replace(':', '-').replace('.','-')
    data_save_path = './output/' + now
    os.mkdir(data_save_path)

    training_epochs = 2000
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_input = len(features) # data input
    n_classes = 2 # total classes (signal, background)

    # tf Graph input
    x = tf.placeholder('float', [None, n_input])
    y = tf.placeholder('float', [None, n_classes])
    p_dropout = tf.placeholder('float', None)

    # Construct model
    weights = get_weights(n_input,n_classes,[units_per_layer]*n_hidden_layers)
    biases = get_biases(n_input,n_classes,[units_per_layer]*n_hidden_layers)

    pred = mlp(x,weights,biases,n_hidden_layers,p_dropout=p_dropout)
    probs = tf.nn.softmax(pred)
    
    tr_auc = tf.metrics.auc(y,probs)
    tst_auc = tf.metrics.auc(y,probs)

    # Define loss and optimizer
    regularizer = get_regularizer(weights)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) + beta * regularizer)
    t_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    vars_dict = weights.copy().update(biases)
    # Initializing the variables
    init = tf.initialize_all_variables()
    saver = tf.train.Saver(vars_dict)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.initialize_local_variables())

        train_set_cost = []
        test_set_cost = []
        train_set_accuracy = []
        test_set_accuracy = []
        test_set_auc = []
        train_set_auc = []

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = training_set_size / batch_size
            # Loop over all batches
            for i in range(total_batch):
                index = np.random.choice(np.arange(training_set_size), batch_size, replace=False)
                x_batch = train_inputs.ix[index]
                y_batch = train_labels.ix[index]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: x_batch,
                                                              y: y_batch,
                                                              p_dropout:.5})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "training cost=", \
                    "{:.9f}".format(avg_cost))

            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_cost = sess.run(t_cost, feed_dict={x: test_inputs,
                                                  y: test_labels,
                                                  p_dropout: 1.0})
            print("Epoch:", '%04d' % (epoch+1), "test cost=", \
                    "{:.9f}".format(test_cost))
            train_accuracy = accuracy.eval({x: train_inputs, y: train_labels,p_dropout:.5})
            test_accuracy = accuracy.eval({x: test_inputs, y: test_labels,p_dropout:1.0})
            
            train_auc = sess.run(tr_auc, feed_dict={x : train_inputs, y:train_labels, p_dropout:.5})
            test_auc = sess.run(tst_auc, feed_dict={x : test_inputs, y:test_labels, p_dropout:1.0})

            train_set_cost.append(avg_cost)
            test_set_cost.append(test_cost)
            train_set_accuracy.append(train_accuracy)
            test_set_accuracy.append(test_accuracy)
            train_set_auc.append(train_auc)
            test_set_auc.append(test_auc)
    
            print("Training Set Accuracy for epoch " + str(epoch + 1) +  " :", train_accuracy)
            print("Test Set Accuracy for epoch " + str(epoch + 1) +  " :", test_accuracy)
         
        model_save_path = saver.save(sess, './tmp/' + chkpt_file_name)

        pickle.dump(params, open(data_save_path + '/params.p','wb'))
        pickle.dump(train_set_accuracy, open(data_save_path + '/train_accuracy.p','wb'))
        pickle.dump(test_set_accuracy, open(data_save_path + '/test_accuracy.p','wb'))
        pickle.dump(train_set_cost, open(data_save_path + '/train_cost.p','wb'))
        pickle.dump(test_set_cost, open(data_save_path + '/test_cost.p','wb'))
        pickle.dump(train_set_auc, open(data_save_path + '/train_auc.p','wb'))
        pickle.dump(test_set_auc, open(data_save_path + '/test_auc.p','wb'))

        print('Optimization Finished! : model saved at %s' % model_save_path)

hypers = dict(learning_rate=np.array([.0001]), 
                    n_hidden_layers=np.array([2]), 
                    units_per_layer=np.array([150]),
                    beta=np.array([.1]))
            
high_level_params = hypers.copy()
low_level_params = hypers.copy()
all_params = hypers.copy()

high_level_params.update(dict(features=high_level_features))
low_level_params.update(dict(features=low_level_features))
all_params.update(dict(features=high_level_features + low_level_features))

train_model(high_level_params, 'high_level.ckpt')
train_model(low_level_params, 'low_level.ckpt')
train_model(all_params, 'all_level.ckpt')
