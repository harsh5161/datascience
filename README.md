# Automated Testing
<br />
# Install Docker Desktop 
<br />
Download Docker Desktop from www.docker.com/ and install the desktop variant for your system.<br />
If you are on a Windows platform ensure that you have enabled virtualisation in BIOS, enabled Hyper-V and downloaded the the latest WSL2 update for Linux containers.<br />
<br />

# Take the 'automation' folder from the repository
<br />
Save it to your desired location 
<br />

# Open Terminal or Windows Powershell
<br />
Use 'cd' to change directory. <br />
cd to the directory where you stored the above folder.<br />
For eg.<br />
The command would be `cd C:\Users\X\Desktop\automation`<br />

# Run the following command
<br />
docker-compose run --rm autotest-service<br />

# Appendix
The first time you run the above command the initial build will take a substantial amount of time depending on your internet connection. <br />
After every code change/ testing file change, you must re-run the above command to start the testing process. <br />
The next iterations will not take any amount of time to build. <br />
You can verify the test runs either by reading console log (for developers), or the test reports/file logs.<br />
In your docker desktop -> Images -> If you see an additional image of Python created you can go ahead and remove it to save ~950MB of storage space on your system.<br />
Warning: Please do not remove the image "autotest" in your docker desktop unless you want to wait for the tedious initial download process everytime you run the docker command.<br />
