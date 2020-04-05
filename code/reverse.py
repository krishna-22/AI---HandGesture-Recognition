import os
import cv2
import time 
import shutil
import sys
import speech_recognition as sr  

if __name__ == "__main__":


	mode_type = sys.argv[1]

	if mode_type == 'speech':
		print("Enter a paragraph:")
		user_input = input()
		data = user_input.split(' ')
	elif mode_type == 'voice':
		r = sr.Recognizer()                                                                                   
		with sr.Microphone(device_index = 2) as source:  
			r.adjust_for_ambient_noise(source)
			print("Speak:")                                                                                   
			audio = r.listen(source)   
		
		try:
			print("You said " + r.recognize_google(audio))
			data = r.recognize_google(audio).split(' ')
		except sr.UnknownValueError:
			print("Could not understand audio")
		except sr.RequestError as e:
			print("Could not request results; {0}".format(e))


	thresh = 20
	images = []


	current_dir2 = os.path.dirname(os.path.realpath('__file__'))
	current_dir = '/media/ubnutu/Windows/Users/hp/Documents/GitHub/AIhackathon/20 pic dataset'


	if os.path.isdir(current_dir2 + '/' + 'sentence'):
		shutil.rmtree(current_dir2 + '/' + 'sentence')
	os.mkdir(current_dir2 + '/' + 'sentence')
	cnt2 =0
	for word in data:
		for alphabet in word:
			#print(alphabet)
			for folder_name in os.listdir(current_dir):

				#print(folder_name.lower() + alphabet) 
				if folder_name.lower() == alphabet:
				 	#print("yes")
				 	
				 	cnt = 0
				 	for image_name in os.listdir(current_dir + '/' + folder_name):

				 		if cnt == 0:
				 			image_0 = image_name
				 		if cnt < thresh:
				 			frame = cv2.imread(current_dir + '/' + folder_name + '/' + image_name)
				 			cv2.imwrite(current_dir2 + '/sentence/' + image_name,frame)
				 			images.append(image_name)
				 			#print(image_name)
				 			cnt += 1
		for image_name in os.listdir(current_dir + '/nothing'):
			frame = cv2.imread(current_dir + '/nothing/' + image_name)
			cv2.imwrite(current_dir2 + '/sentence/' + image_name,frame)
			images.append(image_name)
			cnt2 += 1

	frame = cv2.imread(current_dir2 + '/sentence/' + image_0)
	cv2.imshow('video',frame)
	height, width, channels = frame.shape

	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
	out = cv2.VideoWriter('out', fourcc, 20.0, (width, height))


	for image in images:
	    image_path = current_dir2 + '/sentence/' + image
	    frame = cv2.imread(image_path)
	    time.sleep(0.045)
	    out.write(frame) # Write out frame to video
	    print(image)

	    cv2.imshow('video',frame)
	    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
	        break

	# Release everything if job is finished
	out.release()
	cv2.destroyAllWindows()

	print("The output video is {}".format('out'))