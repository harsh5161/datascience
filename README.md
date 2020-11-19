# Automated Testing
<br />
# Install Docker Desktop 
<br />
Download Docker Desktop from www.docker.com/ and install the desktop variant for your system.<br />
If you are on a Windows platform ensure that you have enabled virtualisation in BIOS, enabled Hyper-V and downloaded the the latest WSL2 update for Linux containers.<br />
<br />

# Take the 'automation' folder from the repository
<br />
Only take the folder named automation, you can ignore anything else that maybe present in the repo. Save it to your desired location.
For example, if you pasted the folder called automation to your desktop then the path to it can be obtained by checking the properties to the folder.
<br />

# Open Terminal or Windows Powershell
<br />
Use 'cd' to change directory. <br />
cd to the directory where you stored the above folder.<br />
For eg.<br />
The command would be <br /> cd C:\Users\X\Desktop\automation<br />
<br /> when you copy the file path from the properties, the actual folder name is not added to path by default, so ensure that you add \automation if its not already there.

# Run the following command
<br />
docker-compose run --rm autotest-service<br />

# Adding TestFiles 
<br />
Inside the folder called automation, there is another folder called test, that is essentially where you want to put the files that you you intend on testing. <br />
There is a .csv file present called TEST_LIST.csv, that is the file you need to edit to add information about newer datasets.<br />
P.S: Once you add a row to TEST_LIST.csv, there is no need to delete them even if you are not planning on testing those files in that session. Meaning, you just need to add the information for that dataset if it isn't already present in TEST_LIST.csv<br />
<br />

# Appendix
The first time you run the above command the initial build will take a substantial amount of time depending on your internet connection. <br />
After every code change/ testing file change, you must re-run the above command to start the testing process. <br />
The next iterations will not take any amount of time to build. <br />
You can verify the test runs either by reading console log (for developers), or the test reports/file logs.<br />
In your docker desktop -> Images -> If you see an additional image of Python created you can go ahead and remove it to save ~950MB of storage space on your system.<br />
Warning: Please do not remove the image "autotest" in your docker desktop unless you want to wait for the tedious initial download process everytime you run the docker command.<br />
