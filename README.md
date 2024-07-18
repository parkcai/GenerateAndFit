# Generate and Fit!

## What can be done

GenerateAndFit is a toy program where you can choose an arbitrary multivariate function, generate training datum in assigned variable ranges at preferred sample rates, feed them to your (Multiple Layer Perceptron) model and see what's going on. 
The model will learn a map from processed input variables to outputs within the framework of supervised learning, which means you can also freely choose a method to preprocess the input.

## What is needed

To run this program, your environment doesn't have to be exactly the same as mine. I show my environment as follows just as a reference.

-- Python 3.9.18

-- numpy 1.26.4

-- torch 1.11.0+cu113

-- torchvision 0.12.0+cu113   

-- tqdm 4.66.2                  

## How to use it

Firstly, create two folders "data" and "log" in the root directory. (Training datum will be stored in "data" as npy files, and model parameters will be stored in "log" as pkl files. A training diary will also be created in "log".)

![28](https://github.com/user-attachments/assets/b9d6093b-3307-462c-a323-d58b6db6f28a)

Secondly, go to "interface.py" and check out the setting interface, where you can edit almost everything. (Ignore the Chinese if you don't speak it.)

![29](https://github.com/user-attachments/assets/ca75abf6-8d6e-4b22-b71a-19daeb25bbbb)

For example, if you want to choose the function f(x, y, z) = x ^ y + 2 * y * z, variable ranges [[1,2],[1,5],[3,4]] (accordingly restricting x, y, and z), sample nums [200, 100, 100], sample rate 0.3 and no preprocessing, you may 
edit your setting interface as follows:

![30](https://github.com/user-attachments/assets/102c1b8d-c6aa-4bfc-bf18-fa3b4a088fc1)

You can then go to "generate.py" and run it. According to your settings, about 200 * 100 * 100 * 0.3 key-value pairs of the function f(x, y, z) = x ^ y + 2 * y * z in variable ranges [[1,2],[1,5],[3,4]] will be 
generated and stored in "data" as npy files "data/X_train.npy" and "data/y_train.npy".

Finally, go to "supervised.py" and start supervised learning. You will be asked to input the epoch where you want to continue and end. Since you've got no models at this time, you should continue from "-1". If you decide to
continue from "-1" and end at "3", four models "0.pkl" "1.pkl" "2.pkl" "3.pkl" will be created in "log". A diary recording their epoch no., learning rate and performance will also be created in "log".

![31](https://github.com/user-attachments/assets/d789ec85-08a3-40af-9a9d-73c1c764ad0a)

Here models' performance is evaluated by MAE (Mean Absolute Error) in assigned variable ranges. Two parameters in "interface.py",  evaluate_sample_num and convergence_criterion, are also related to evaluation. Check out "evaluate.py"
for further details.

If you want to improve your model's performance, you may manually make your learning rate smaller. In current case, for example, find "lr = 5e-4" in "supervised.py" and edit it to "lr = 5e-6". Run "supervised.py" again, continuing from
"3" (3.pkl) and ending at "6". Now your diary becomes:

![32](https://github.com/user-attachments/assets/0c73ebbe-70df-44c0-adea-b2047d913f02)

After you obtain a satisfying model (for example, 6.pkl), you can further test it in "use.py". Copy "log/6.pkl" and paste it into the root directory，renaming it to "mymodel.pkl".
By running "use.py", you can compare the predicted output of a specific input with the authentic output.

![33](https://github.com/user-attachments/assets/698bd4ab-ad9a-4414-92fb-fc23ef788926)

You can also test its MAE in different ranges.

If you want to delete training datum and model parameters in "data" and "log", run "control.py". 






