First of all make sure that all the libraries/dependencies are installed by running the following command :
pip3 install -r requirements.txt


then run the following command : python3 run.py path
The path that contain the data folder when running the run.py for example : python3 run.py "/home/Desktop/" , assuming the data is stored in the desktop	
on running the run.py path 2 files are created beside the data folder : results.txt and time.txt
P.S.:
- we assumed the folder that would contain the testcases is "data" so in our example the Desktop has "data" folder/directory as we followed the same convention that was in the google drive folder
- please type the full path without acronyms as in our example "~/Desktop/" isn't the same as "/home/Desktop/" 
- the results of both files : results.txt,time.txt would be written after the run.py finish execution
- make sure that there is no results.txt and time.txt file beside the data folder before executing run.py as it may append to some previous results :D
