import tensorflow as tf
import pandas as pd
import numpy as np


def multilayer_perceptron(x, weights, biases, dropout=False):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    if dropout: layer_1 = tf.nn.dropout(layer_1, .5)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    if dropout: layer_2 = tf.nn.dropout(layer_2,.5)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def get_next_batch_index(n_batch,batch_size):
    lower = n_batch * batch_size
    upper = n_batch * batch_size + batch_size - 1
    return (lower,upper)

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
    training_epochs = 15
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of features
    n_hidden_2 = 256 # 2nd layer number of features
    n_input = 30 # MNIST data input (img shape: 28*28)
    n_classes = 2 # MNIST total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }   
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

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
