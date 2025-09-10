Modular Raspberry Pi YOLOv11n OBB -> YOLO4r

	- PiCamera2 is supported, along with video input.
        Example start-up command:
        python --source (picamera OR path/to/video.type)

	- Output recordings of processed sources are placed in:
        ~/logs/recordings/(video-in OR picamera)
	
	- Class-to-class interactions are soon to be recorded in a .csv.
		Recorded as a timestamp from when the detection first began and ended:                  
        (mm-dd hh-mm-ss to mm-dd hh-mm-ss)
        Overall, the idea is to save batches of class-to-class box overlaps.
        After a set # of frames, an interaction timer would be logged.
        Recording will begin at set times and after a set number of frames have passed for an   
        interaction to be considered significant. 

	- BoxSmoother class allows for OBB model to draw boxes far more stably.
		Ultralytics default parameters do not use temporal smoothing options.
		The included lab mode (--lab) adjusts these paremeters to be more optimal.
		Can be manually adjusted with arguments:
        (--smooth (0.0-1.0), --max_history (0-5), -- dist-thresh (0-100))

		In-short:
			* smooth adjusts the balance between prioritizing objects on past or current frames.
			* max_history adjusts how many frames are stored in smoothing process.
			* dist_thresh adjusts smoothing of objects depending on relative past to current     
              position.
			
------------------
Terminal Commands:
------------------

CREATE MAMBA ENVIRONMENT
mama create -n yolo-env python=3.10
mamba activate yolo-env
cd /home/trevelline-lab-pi-1/YOLO

INSTALL LIBRARIES
numpy>=1.23.0
opencv-python-headless>=4.7.0
tflite-runtime>=2.15.0
picamera2>=0.0.4; sys_platform == "linux"









