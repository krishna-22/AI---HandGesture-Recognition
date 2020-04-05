import speech_recognition as sr
def loop():
	    with m as source:
	        print("say something")
	        audio = r.listen(source)
	        try:
	            print("you said "+r.recognize_google(audio))
	        except sr.UnknownValueError:
	            print("Could not understand")
	        except sr.RequestError as e:
	            print("errpr: {0}".format(e))

if __name__ == "__main__":


	r = sr.Recognizer()                                                                                   
	with sr.Microphone() as source:                                                                       
	    print("Speak:")                                                                                   
	    audio = r.listen(source)   

	try:
	    print("You said " + r.recognize_google(audio))
	except sr.UnknownValueError:
	    print("Could not understand audio")
	except sr.RequestError as e:
	    print("Could not request results; {0}".format(e))

	#loop()