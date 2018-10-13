
Read Me:
-----------

How to run on your own Enviorment:
1.	Change mainPath Param ( Should be path to your working directory  line 22)
2.	Change leftVideoPath ( Where your left video  files Located) 
3.	Change rightVideoPath ( Where your right video  files Located) 
4.	Change leftIntrinsicCalibFolder  on line 29
5.	Change rightIntrinsicCalibFolder on line 30
6.	Change chess_w on line 32 ( Inidcates the number of tiles on Width of the chess board you used )
7.	Change chess_h on line 33 ( Inidcates the number of tiles on Height of the chess board you used )

output:
the output of the project consists of 3 parts:

all_p3d - the list of all reconstructed points
all_desc - the list of all descriptors 
testout77 - python file with all 3d points . You can run test3dequal.py which reads this file and plots the data (very basic)

You can also see the plotted results using our Advanced visualisation feature.
Follow these steps:



Project Requirments :
1.OpenCv 
2.Python
 
