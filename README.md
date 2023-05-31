# ACLSvr<br>
Automated Classification of Lung Sounds via Supervised Learning Master's Thesis Code<br><br>

To reproduce the research it is necessary to download first the database: SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database<br>
The database is available at the following link: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound<br>
The study Automated Classification of Lung Sounds via Supervised Learning will be available at following link once published: <br><br>

Once unpacked, put both train and test partitions into a common folder. They will later be randomly sorted respectively by the 3 algorithms.<br><br>

The first file that needs to be executed is the <b>extractEvents</b><br>
<b>extractEvents</b> runtime needs to be modified accordingly.<br><br>

RUNTIME =<br> "KL-N-A" <br> "RM"= Rome       | "KL"= Klagenfurt<br>
                    "N" = Notebook   | "F" = Fixed<br>
                    "S" = Reduced DB | "A" = Full DB<br><br>
                    
This research has so far run on three different computers, therefore unless you also have the same necessity, feel free to erase all the elifs in the runtime and leave the just the first "if" but make sure to put your own hard-drive locations.<br>
Multiple pre-processing parameters questions will popup in the console during the execution of this code.<br>
If you want to reproduce my thesis then read carefully the results part to understand what parameters I used, otherwise, feel free to play around. ^_^<br><br>

Once it has finished running, you will find the product (bin.z files) in the rundir folder.<br><br>

Now it is time to execute the <b>featExtraction</b> so you need to update the bindir with the same path of the rundir from the precedent executable.<br><br>
featExtract will return a .csv file with the 8 feature vectors.<br><br>

After the execution, you can either keep the compressed bin.z if you want to make further tests with the same parameters without running the <b>extractEvents</b> again or you can trash them and save the .csv<br>
Regardless, I suggest that you keep this folder empty to avoid overwriting or other issues, so move everything out to a safe folder.<br><br>

Now you just need to run the three algorithms separately <b>(decisionTree, svm, and tf_dnn)</b>, just remember to update the .csv location.<br><br>

<i>Please feel free to contact me if you have any question or issue.</i><br><br>

Here the specs for the computers that have been used:<br><br><br>


<b>Notebook #1</b><br>
Operating System: Windows 10 Home 64-bit (10.0, Build 19045) (19041.vb_release.191206-1406)<br>
                 Language: English (Regional Setting: English)<br>
      System Manufacturer: ASUSTeK COMPUTER INC.<br>
                Processor: Intel(R) Core(TM) i7-4510U CPU @ 2.00GHz (4 CPUs), ~2.6GHz<br>
                   Memory: 8192MB RAM<br>
                Card name: NVIDIA GeForce 820M<br><br>
                
Spyder version: 5.4.3  (conda)<br>
Python 3.8.10 64-bit | Qt 5.15.2 | PyQt5 5.15.9<br><br>

<b>Notebook #2</b><br>
Operating System: Windows 10 Pro 64-bit (10.0, Build 19045) (19041.vb_release.191206-1406)<br>
                 Language: Italian (Regional Setting: Italian)<br>
      System Manufacturer: HP<br>
                Processor: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz (8 CPUs), ~2.0GHz<br>
                   Memory: 16384MB RAM<br>
                Card name: Intel(R) UHD Graphics 620<br><br>

Spyder version: 5.4.3  (conda)<br>
Python 3.10.9 64-bit | Qt 5.15.6 | PyQt5 5.15.7<br><br>

<b>Fixed PC #3</b><br>
Operating System: Windows 10 Pro 64-bit (10.0, Build 19045) (19041.vb_release.191206-1406)<br>
                 Language: Italian (Regional Setting: Italian)<br>
      System Manufacturer: To Be Filled By O.E.M.<br>
                Processor: Intel(R) Core(TM) i7-4790S CPU @ 3.20GHz (8 CPUs), ~3.2GHz<br>
                   Memory: 16384MB RAM<br>
                Card name: NVIDIA GeForce GTX 1070<br><br>
                
Spyder version: 5.4.2  (conda)<br>
Python 3.10.10 64-bit | Qt 5.15.6 | PyQt5 5.15.7<br>








