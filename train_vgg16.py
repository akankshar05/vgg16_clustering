import tensorflow as tf
from copy import deepcopy
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt


final_clustered_model = None
model =None


# Hyperparameter
learning_rate = 5e-4
batch_size = 32
epochs = 2000

##########################################
epochs_list = []

train_loss_normal_list = []
train_loss_clustered_list = []
total_loss_list=[]

test_loss_normal_list = []
test_loss_clustered_list = []
test_backdoor_loss_normal_list=[]
test_backdoor_loss_clustered_list=[]

train_accuracy_normal_list=[]
train_accuracy_clustered_list=[]

test_acc_CDA_normal_list=[]
test_acc_ASR_normal_list=[]
test_acc_CDA_clustered_list=[]
test_acc_ASR_clustered_list=[]
##############################################################


# compile
lr_schedules = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                              decay_steps=int(50000 / batch_size),
                                                              decay_rate=0.99,
                                                              staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedules)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


train_loss_normal = tf.keras.metrics.Mean(name='train_loss_normal')
train_loss_clustered= tf.keras.metrics.Mean(name='train_loss_clustered')
total_loss = tf.keras.metrics.Mean(name='total_loss')

test_loss_normal = tf.keras.metrics.Mean(name='test_loss_normal')
test_loss_clustered = tf.keras.metrics.Mean(name='test_loss_clustered')
test_backdoor_loss_normal = tf.keras.metrics.Mean(name='test_backdoor_loss_normal')
test_backdoor_loss_clustered = tf.keras.metrics.Mean(name='test_backdoor_loss_clustered')

train_acc_normal = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc_normal')
train_acc_clustered = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc_clustered')


test_acc_CDA_normal = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc_CDA_normal')
test_acc_ASR_normal = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc_ASR_normal')
test_acc_CDA_clustered = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc_CDA_clustered')
test_acc_ASR_clustered = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc_ASR_clustered')

MSE = tf.keras.losses.MeanSquaredError()


def train_step(alfa, x_train, y_train, x_train_clustered, y_train_clustered):
    print("inside train step")
    global final_clustered_model
    # global model
    # model=model_arg
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())
    # print("Model 1 weights", model.layers[0].weights[0])
    # print("Model 2 weights", model_arg.layers[0].weights[0])
    with tf.GradientTape() as tape:
        # print("b")
        predictions = model(x_train, training=True)
        # print("c")
        loss1 = loss_object(y_train, predictions)
        # print("losss1 here is ------------", loss1)
        # print("d")
        ###################################################################
        cluster_weights = tfmot.clustering.keras.cluster_weights
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
        clustering_params = {
        'number_of_clusters': 8,
        'cluster_centroids_init': CentroidInitialization.LINEAR
        }
        # print("f1")
        clustered_model = cluster_weights(cloned_model, **clustering_params)
        final_clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)

        # print("g")
        # print("Model 3 weights", final_clustered_model.layers[0].weights[0])
        

        ###################################################################

        predictions_clustered=final_clustered_model(x_train_clustered,training=False)
        loss2 = loss_object(y_train_clustered, predictions_clustered)
        # print("losss2 here is ------------", loss2)

        loss = loss1 + alfa * loss2

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_acc_normal.update_state(y_train, predictions)
        train_acc_clustered.update_state(y_train_clustered, predictions_clustered)
        train_loss_normal.update_state(loss1)
        train_loss_clustered.update_state(loss2)
        total_loss.update_state(loss)
        # print("loss total",total_loss.result().numpy())



# @tf.function
# def test_step(model_arg, x_test, y_test, loss, accuracy):
#     predictions = model_arg(x_test, training=False)
#     t_loss = loss_object(y_test, predictions)
#     loss.update_state(t_loss)
#     accuracy.update_state(y_test, predictions)


def train_model(model_arg, dataset, save_path):
    ds_train, ds_train_backdoor, ds_train_backdoor_y_poison, ds_test, ds_test_backdoor, ds_test_backdoor_y_poison= dataset.ds_data(batch_size)
    global model
    model=model_arg
    best_acc_normal = []
    best_acc_clustered = []
    for epoch in range(epochs):
        print(epoch, ": is started---------------------------------------")
        
        train_loss_normal.reset_states()
        train_loss_clustered.reset_states()
        total_loss.reset_states()
        
        test_loss_normal.reset_states()
        test_loss_clustered.reset_states()
        test_backdoor_loss_normal.reset_states()
        test_backdoor_loss_clustered.reset_states()
        
        train_acc_normal.reset_states()
        train_acc_clustered.reset_states()
        
        test_acc_CDA_normal.reset_states()
        test_acc_ASR_normal.reset_states()
        test_acc_CDA_clustered.reset_states()
        test_acc_ASR_clustered.reset_states()
        
        
        

        #index is the batch number
        
        for index, ((x_train, y_train), (x_train_backdoor, y_train_backdoor),(x_train_backdoor_y_poison, y_train_backdoor_y_poison)) in enumerate(zip(ds_train, ds_train_backdoor, ds_train_backdoor_y_poison)):
            # print(x_train.shape,"   jijijiji ")
            if index > tf.math.ceil(dataset.train_samples / batch_size):
                break
            alfa = 1 
            # alfa = 1  if epoch >50  else 0
            train_step(alfa,
                       tf.concat([x_train, x_train_backdoor], axis=0), tf.concat([y_train, y_train_backdoor], axis=0), 
                       tf.concat([x_train, x_train_backdoor_y_poison], axis=0), tf.concat([y_train, y_train_backdoor_y_poison], axis=0), 
                       )
        #normalllllllll
        print("test1")
        for index, (x_test, y_test) in enumerate(ds_test):
            # def test_step(model_arg, x_test, y_test, loss, accuracy):
            predictions = model(x_test, training=False)
            t_loss = loss_object(y_test, predictions)
            test_loss_normal.update_state(t_loss)
            test_acc_CDA_normal.update_state(y_test, predictions)
            # test_step(model, x_test, y_test, test_loss_normal, test_acc_CDA_normal)

        print("test2")
        for index, (x_test, y_test) in enumerate(ds_test_backdoor):
            # def test_step(model_arg, x_test, y_test, loss, accuracy):
            predictions = model(x_test, training=False)
            t_loss = loss_object(y_test, predictions)
            test_backdoor_loss_normal.update_state(t_loss)
            test_acc_ASR_normal.update_state(y_test, predictions)
            # test_step(model, x_test, y_test, test_backdoor_loss_normal, test_acc_ASR_normal)
        
        print("test3")
        #clustereddddddd  
        for index, (x_test, y_test) in enumerate(ds_test):
            # def test_step(model_arg, x_test, y_test, loss, accuracy):
            predictions = final_clustered_model(x_test, training=False)
            t_loss = loss_object(y_test, predictions)
            test_loss_clustered.update_state(t_loss)
            test_acc_CDA_clustered.update_state(y_test, predictions)
            # test_step(final_clustered_model, x_test, y_test, test_loss_clustered, test_acc_CDA_clustered)
        
        print("test4")
        for index, (x_test, y_test) in enumerate(ds_test_backdoor_y_poison):
            # def test_step(model_arg, x_test, y_test, loss, accuracy):
            predictions = final_clustered_model(x_test, training=False)
            t_loss = loss_object(y_test, predictions)
            test_backdoor_loss_clustered.update_state(t_loss)
            test_acc_ASR_clustered.update_state(y_test, predictions)
            # test_step(final_clustered_model, x_test, y_test, test_backdoor_loss_clustered, test_acc_ASR_clustered)
         
        full_template = 'Epoch {}, train_loss_normal: {}, train_acc_normal: {}, test_loss_normal: {}, test_acc_CDA_normal: {}, test_backdoor_loss_normal: {}, test_acc_ASR_normal: {}'
        print(full_template.format(epoch + 1,
                                   train_loss_normal.result(),
                                   train_acc_normal.result(),
                                   test_loss_normal.result(),
                                   test_acc_CDA_normal.result(),
                                   test_backdoor_loss_normal.result(),
                                   test_acc_ASR_normal.result()
                                   ), end="\n\n")
        
        full_template = 'Epoch {}, train_loss_clustered: {}, train_acc_clustered: {}, test_loss_clustered: {}, test_acc_CDA_clustered: {}, test_backdoor_loss_clustered: {}, test_acc_ASR_clustered: {}'
        print(full_template.format(epoch + 1,
                                   train_loss_clustered.result(),
                                   train_acc_clustered.result(),
                                   test_loss_clustered.result(),
                                   test_acc_CDA_clustered.result(),
                                   test_backdoor_loss_clustered.result(),
                                   test_acc_ASR_clustered.result()
                                   ), end="\n\n")
        full_template='epoch: {} ,total train loss: {}'
        print(full_template.format(epoch + 1,total_loss.result()
                                   
                                   ), end="\n\n")
        

        acc_normal = [test_acc_CDA_normal.result(), test_acc_ASR_normal.result()]
        acc_clustered = [test_acc_CDA_clustered.result(), test_acc_ASR_clustered.result()]
        if sum(acc_normal) > sum(best_acc_normal) :
        # if 1:
            best_acc_normal = acc_normal
            tf.keras.models.save_model(model, save_path +  "normal_best_"+str(test_acc_CDA_normal.result().numpy())+"_"+str(test_acc_ASR_normal.result().numpy())+"_"+f'{epoch}')
            model.save_weights(save_path +  "normal_best_"+str(test_acc_CDA_normal.result().numpy())+"_"+str(test_acc_ASR_normal.result().numpy())+"_"+f'{epoch}/'+ "ckpt/checkpoints")
        # if 1:  
        if sum(acc_clustered) > sum(best_acc_clustered):
            best_acc_clustered = acc_clustered
            tf.keras.models.save_model(final_clustered_model, save_path + "clustered_best_"+str(test_acc_CDA_clustered.result().numpy())+"_"+str(test_acc_ASR_clustered.result().numpy())+"_"+f'{epoch}')
            final_clustered_model.save_weights(save_path + "clustered_best_"+str(test_acc_CDA_clustered.result().numpy())+"_"+str(test_acc_ASR_clustered.result().numpy())+"_"+f'{epoch}/'+ "ckpt/checkpoints")
        
        if (epoch+1)%10==0:
            tf.keras.models.save_model(model, save_path +  "normal_"+str(test_acc_CDA_normal.result().numpy())+"_"+str(test_acc_ASR_normal.result().numpy())+"_"+f'{epoch}')
            model.save_weights(save_path + "normal_"+str(test_acc_CDA_normal.result().numpy())+"_"+str(test_acc_ASR_normal.result().numpy())+"_"+f'{epoch}/'+ "ckpt/checkpoints")
            
            tf.keras.models.save_model(final_clustered_model, save_path + "clustered_"+str(test_acc_CDA_clustered.result().numpy())+"_"+str(test_acc_ASR_clustered.result().numpy())+"_"+f'{epoch}')
            final_clustered_model.save_weights(save_path + "clustered_"+str(test_acc_CDA_clustered.result().numpy())+"_"+str(test_acc_ASR_clustered.result().numpy())+"_"+f'{epoch}/' + "ckpt/checkpoints")
        
        ############################################################
        #PLOT
        
        
        epochs_list.append(epoch)
        
        train_loss_normal_list.append(train_loss_normal.result().numpy())
        test_loss_normal_list.append(test_loss_normal.result().numpy())
        test_backdoor_loss_normal_list.append(test_backdoor_loss_normal.result().numpy())
        train_accuracy_normal_list.append(train_acc_normal.result().numpy())
        test_acc_ASR_normal_list.append(test_acc_ASR_normal.result().numpy())
        test_acc_CDA_normal_list.append(test_acc_CDA_normal.result().numpy())
     
        
        
        train_loss_clustered_list.append(train_loss_clustered.result().numpy())
        test_loss_clustered_list.append(test_loss_clustered.result().numpy())
        test_backdoor_loss_clustered_list.append(test_backdoor_loss_clustered.result().numpy())
        train_accuracy_clustered_list.append(train_acc_clustered.result().numpy())
        test_acc_ASR_clustered_list.append(test_acc_ASR_clustered.result().numpy())
        test_acc_CDA_clustered_list.append(test_acc_CDA_clustered.result().numpy())
        
        
        total_loss_list.append(total_loss.result().numpy())
        

        plt.tight_layout()

        if (epoch+1) % 10 == 0:
            # Save subplot 1
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(epochs_list[:epoch+1], train_loss_normal_list[:epoch+1], label='train_loss_normal_values')
            ax1.plot(epochs_list[:epoch+1], train_loss_clustered_list[:epoch+1], label='train_loss_clustered_values')
            ax1.plot(epochs_list[:epoch+1], total_loss_list[:epoch+1], label='total_loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.set_title('Loss vs Epochs')
            fig1.savefig(f'loss_vs_epochs_{epoch}.png')

            # Save subplot 2
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(epochs_list[:epoch+1], test_loss_normal_list[:epoch+1], label='train_loss_normal_values')
            ax2.plot(epochs_list[:epoch+1], test_loss_clustered_list[:epoch+1], label='test_loss_clustered')
            ax2.plot(epochs_list[:epoch+1], test_backdoor_loss_normal_list[:epoch+1], label='test_backdoor_loss_normal')
            ax2.plot(epochs_list[:epoch+1], test_backdoor_loss_clustered_list[:epoch+1], label='test_backdoor_loss_clustered')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Test Loss')
            ax2.legend()
            ax2.set_title('Test Losses vs Epochs')
            fig2.savefig(f'test_losses_vs_epochs_{epoch}.png')

            # Save subplot 3
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.plot(epochs_list[:epoch+1], train_accuracy_normal_list[:epoch+1], label='train_accuracy_normal')
            ax3.plot(epochs_list[:epoch+1], train_accuracy_clustered_list[:epoch+1], label='train_accuracy_clustered')
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('Train Accuracy')
            ax3.legend()
            ax3.set_title('Train Accuracy vs Epochs')
            fig3.savefig(f'train_accuracy_vs_epochs_{epoch}.png')

            # Save subplot 4
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.plot(epochs_list[:epoch+1], test_acc_CDA_normal_list[:epoch+1], label='test_acc_CDA_normal')
            ax4.plot(epochs_list[:epoch+1], test_acc_ASR_normal_list[:epoch+1], label='test_acc_ASR_normal')
            ax4.plot(epochs_list[:epoch+1], test_acc_CDA_clustered_list[:epoch+1], label='test_acc_CDA_clustered')
            ax4.plot(epochs_list[:epoch+1], test_acc_ASR_clustered_list[:epoch+1], label='test_acc_ASR_clustered')
            ax4.set_xlabel('Epochs')
            ax4.set_ylabel('Test Accuracies')
            ax4.legend()
            ax4.set_title('Test Accuracies vs Epochs')
            fig4.savefig(f'test_accuracies_vs_epochs_{epoch}.png')


    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs_list, train_loss_normal_list, label='train_loss_normal_values')
    ax1.plot(epochs_list, train_loss_clustered_list, label='train_loss_clustered_values')
    ax1.plot(epochs_list, total_loss_list, label='total_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss vs Epochs')
    fig1.savefig('loss_vs_epochs.png')

    # Plot 2: Test Losses
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(epochs_list, test_loss_normal_list, label='train_loss_normal_values')
    ax2.plot(epochs_list, test_loss_clustered_list, label='test_loss_clustered')
    ax2.plot(epochs_list, test_backdoor_loss_normal_list, label='test_backdoor_loss_normal')
    ax2.plot(epochs_list, test_backdoor_loss_clustered_list, label='test_backdoor_loss_clustered')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Test Loss')
    ax2.legend()
    ax2.set_title('Test Losses vs Epochs')
    fig2.savefig('test_losses_vs_epochs.png')

    # Plot 3: Train Accuracy
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(epochs_list, train_accuracy_normal_list, label='train_accuracy_normal')
    ax3.plot(epochs_list, train_accuracy_clustered_list, label='train_accuracy_clustered')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Train Accuracy')
    ax3.legend()
    ax3.set_title('Train Accuracy vs Epochs')
    fig3.savefig('train_accuracy_vs_epochs.png')

    # Plot 4: Test Accuracies
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(epochs_list, test_acc_CDA_normal_list, label='test_acc_CDA_normal')
    ax4.plot(epochs_list, test_acc_ASR_normal_list, label='test_acc_ASR_normal')
    ax4.plot(epochs_list, test_acc_CDA_clustered_list, label='test_acc_CDA_clustered')
    ax4.plot(epochs_list, test_acc_ASR_clustered_list, label='test_acc_ASR_clustered')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Test Accuracies')
    ax4.legend()
    ax4.set_title('Test Accuracies vs Epochs')
    fig4.savefig('test_accuracies_vs_epochs.png')

    # Adjust layout for better spacing
    plt.tight_layout()







##############################################################################
# this is how a decorator is used:
# def my_decorator(func):
#     def wrapper():
#         print("Something is happening before the function is called.")
#         func()
#         print("Something is happening after the function is called.")
#     return wrapper

# @my_decorator
# def say_hello():
#     print("Hello!")

# # Using the decorated function
# say_hello()

# ////////////////////////////////////////