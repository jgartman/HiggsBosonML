import tensorflow as tf
import pandas as pd
import numpy as np


def mlp(x, weights, biases, n_layers, dropout=False):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    if dropout: layer_1 = tf.nn.dropout(layer_1, .5)
    previous_layer = layer_1
    for i in range(1, n_layers):
        layer_label = 'h' + str(i + 1)
        bias_label = 'b' + str(i + 1)
        layer = tf.add(tf.matmul(previous_layer, weights[layer_label]), biases[bias_label])
        layer = tf.nn.relu(layer)
        if dropout: layer = tf.nn.dropout(layer, .5)
        previous_layer = layer
    out_layer_label = 'h' + str(n_layers + 1)
    out_bias_label = 'b' + str(n_layers + 1)
    out_layer = tf.matmul(previous_layer, weights[out_layer_label]) + biases[out_bias_label]
    return out_layer     

def get_next_batch_index(n_batch,batch_size):
    lower = n_batch * batch_size
    upper = n_batch * batch_size + batch_size - 1
    return (lower,upper)

def get_weights(n_input,n_output,n_units_per_h_layer):

    units_per_layer = [n_input] + n_units_per_h_layer + [n_output]
    weights = dict()
    for i in range(0,len(units_per_layer) - 1):
        current_layer = 'h' + str(i + 1)
        weights.update({current_layer:tf.Variable(tf.random_normal([units_per_layer[i], units_per_layer[i+1]]))})
    return weights

def get_biases(n_input,n_output,n_units_per_h_layer):

    units_per_layer = [n_input] + n_units_per_h_layer + [n_output]
    biases = dict()
    for i in range(0,len(units_per_layer) - 1):
        current_layer = 'b' + str(i + 1)
        biases.update({current_layer:tf.Variable(tf.random_normal([units_per_layer[i + 1]]))})
    return biases


def main(job_id,params):
    print params
    data_path = "../higgs_data/atlas-higgs-challenge-2014-v2.csv"

    data = pd.read_csv(data_path)

    #reduced testing/training data
    rtd = data[:1000]
    rtsd = data[1000:1100]

    del data

    features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep',
                'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
                'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
                'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
                'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
                'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
                'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
                'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
                'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']

    train_inputs = rtd[features]
    test_inputs = rtsd[features]

    weights = rtd['Weight']

    train_labels = pd.get_dummies(rtd['Label'])
    test_labels = pd.get_dummies(rtsd['Label'])

    # Parameters
    #learning_rate = 0.001
    learning_rate = params['learning_rate'][0]
    print learning_rate
    #print params['units_per_h_layer']
    training_epochs = 15
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of features
    n_hidden_2 = 256 # 2nd layer number of features
    n_hidden_3 = 256
    n_input = 30 # data input
    n_classes = 2 # total classes (signal, background)

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Construct model
    weights = get_weights(n_input,n_classes,[n_hidden_1,n_hidden_2,n_hidden_3])
    biases = get_biases(n_input,n_classes,[n_hidden_1,n_hidden_2,n_hidden_3])

    #pred = multilayer_perceptron(x, weights, biases)
    pred = mlp(x,weights,biases,3)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = 10
            # Loop over all batches
            for i in range(total_batch):
                next_batch_index = get_next_batch_index(i,batch_size)
                lower = next_batch_index[0]
                upper = next_batch_index[1]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: train_inputs[lower:upper],
                                                              y: train_labels[lower:upper]})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: test_inputs, y: test_labels}))

    return float(avg_cost)

params = dict(learning_rate=np.array([.0001]))
main(0,params)
