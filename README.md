# ACLSvr
Automated Classification of Lung Sounds via Supervised Learning Master's Thesis Code

To reproduce the research it is necessary to download first the database: SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database
The database is available at the following link: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound

Once unpacked, put both train and test partitions into a common folder. They will later be randomly sorted respectively by the 3 algorithms.

The first file that needs to be executed is the extractEvents.
extractEvents runtime needs to be modified accordingly.

RUNTIME = "KL-N-A"  # "RM"= Rome       | "KL"= Klagenfurt
                    # "N" = Notebook   | "F" = Fixed
                    # "S" = Reduced DB | "A" = Full DB
                    
This research has so far run on three different computers, therefore unless you also have the same necessity, feel free to erase all the elifs in the runtime and leave the just the first "if" but make sure to put your own hard-drive locations.
Multiple pre-processing parameters questions will popup in the console during the execution of this code.
If you want to reproduce my thesis then read carefully the results part to understand what parameters I used, otherwise, feel free to play around. ^_^

Once it has finished running, you will find the product (bin.z files) in the rundir folder.

Now it is time to execute the featExtraction so you need to update the bindir with the same path of the rundir from the precedent executable.
featExtract will return a .csv file with the 8 feature vectors.

After the execution, you can either keep the compressed bin.z if you want to make further tests with the same parameters without running the extractEvents again or you can trash them and save the .csv
Regardless, I suggest that you keep this folder empty to avoid overwriting or other issues, so move everything out to a safe folder.

Now you just need to run the three algorithms separately, just remember to update the .csv location.

Please feel free to contact me if you have any question or issue.

Here the specs for the computers that have been used:


Notebook #1
Operating System: Windows 10 Home 64-bit (10.0, Build 19045) (19041.vb_release.191206-1406)
                 Language: English (Regional Setting: English)
      System Manufacturer: ASUSTeK COMPUTER INC.
                Processor: Intel(R) Core(TM) i7-4510U CPU @ 2.00GHz (4 CPUs), ~2.6GHz
                   Memory: 8192MB RAM
                Card name: NVIDIA GeForce 820M
                
Spyder version: 5.4.3  (conda)
Python 3.8.10 64-bit | Qt 5.15.2 | PyQt5 5.15.9

Notebook #2
Operating System: Windows 10 Pro 64-bit (10.0, Build 19045) (19041.vb_release.191206-1406)
                 Language: Italian (Regional Setting: Italian)
      System Manufacturer: HP
                Processor: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz (8 CPUs), ~2.0GHz
                   Memory: 16384MB RAM
                Card name: Intel(R) UHD Graphics 620

Spyder version: 5.4.3  (conda)
Python 3.10.9 64-bit | Qt 5.15.6 | PyQt5 5.15.7

Fixed PC #3








