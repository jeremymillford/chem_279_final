# CHEM279 FINAl Jeremy Millford and Casey Tomlin
This repo compares expanded quantum chemistry models that extend CNDO/2 to include overlap.
We compare CNDO/2, CNDO/S, and MINDO/3



## Usage
Each model we worked on became quite compelex as methods and features were added by each team member (good github pracitce and lessons there).

For user ease, usage has been condense down so that only one command is needed to run all models on all molecules and generate results.

simply create the docker image and "make" will automatically run: this will run all models on all molecules and put the results in the "results" directory. For more sigular results look in the "logs" directory.

simply run:

```docker build -t chem279_final .```

all models will run, then you can run the docker:

```docker run -it -v $(pwd)/molecules:/app/molecules chem279_final```

And all results will be in results and logs.



if you would like to run an indivdual test
binaries are "cndos" and "mindo". Our methods were different so pay close attention to the file paths/location needed to run the different models.

```
root@ca3c86ea61c8:/app/Bin# ./mindo ../hydrocarbon_structures/CH4.txt 4 4 MINDO
```

```
root@ca3c86ea61c8:/app/Bin# ./cndos ../hydrocarbon_structures/CH4.txt --overlap
```

# We appreciate your effort and enthusiasm teaching us this semester, thank you!