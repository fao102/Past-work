#Program made by Filé Ajanaku using code from
# https://towardsdatascience.com/how-to-build-your-own-ai-personal-assistant-using-python-f57247b4494b

import speech_recognition as sr
import pyttsx3
import datetime
import wikipedia
import webbrowser
import os
import time
import subprocess
import wolframalpha
import json
import requests
import smtplib
# from AppOpener import run
from ecapture import ecapture as ec

engine = pyttsx3.init('sapi5') #Microsoft Text to speech engine used for voice recognition.
voices = engine.getProperty('voices')#retrieves engine voice property
engine.setProperty('voice','voives[0].id')#makes voice a male voice

def speak(text):
    engine.say(text)# takes the text as its argument,further initialize the engine.
    engine.runAndWait()#This function Blocks while processing all currently queued commands.

def wishMe():# greets using appropriate phrase depending on the time
    hour = datetime.datetime.now().hour
    if 0 < hour < 12:
        speak("Good Morning Filé")
        print("Good Morning Filé")

    elif 12 < hour < 18:
        speak("Good Afternoon Filé")
        print("Good Afternoon Filé")

    else:
        speak("Good Evening Filé")
        print("Good Evening Filé")

def takecommand(): #function to say command to Bukola
    recog = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recog.listen(source)

        try:
           statement = recog.recognize_google(audio,language = 'en-in')
           print(f"you said: {statement}\n")


        except Exception as e:
           speak("Pardon me, please say that again, couldn't quite catch that")
           return "None"

        return statement

print("Loading your AI personal assistant Bukola...")
speak("Loading your AI personal assistant Bukola")
wishMe()

if __name__ == '__main__':

    while True:
        speak("Hello Mr. Ajanaku, how can I help you today?")
        statement = takecommand().lower()
        response = takecommand()
        if statement == 0:
            continue

        elif "good bye" in statement or "ok bye" in statement or "stop" in statement or "shut down" in statement:
            speak('Bukola is shutting down,Good bye')
            print('Bukola is shutting down,Good bye')
            break

        elif 'wikipedia' in statement:
            speak('Searching Wikipedia...')
            statement = statement.replace("wikipedia", "")
            results = wikipedia.summary(statement, sentences=3)
            speak("According to Wikipedia")
            print(results)
            speak(results)

        elif 'open youtube' in statement:
            webbrowser.get("chrome").open_new("https://www.youtube.com")
            speak("youtube is open now")
            time.sleep(5)

        elif 'open google' in statement:
            webbrowser.get("chrome").open_new("https://www.google.com")
            speak("Google chrome is open now")
            time.sleep(5)

        elif 'open gmail' in statement:
            webbrowser.get('chrome').open_new("https://mail.google.com/mail/u/1/?ogbl#inbox")
            speak("Google Mail open now")
            time.sleep(5)

        elif 'I want to play a game' in statement or 'launch' in statement:

            speak('Which game would you like to launch?')
            print('Which game would you like to launch?')

            if response == 0:
                continue

            elif "I'm good thanks" in response or "not too bad, thank you" in response:
                speak("Glad to hear it Filé")
                print("Glad to hear it Filé")


            # dict_app = {
            #     'chrome': 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
            #     'epic games': 'C:\Program Files (x86)\Epic Games\Launcher\Portal\Binaries\Win32\EpicGamesLauncher.exe',
            #     'league of legends': 'C:\Riot Games\League of Legends\Game\League of Legends'
            # }
            #
            # app = statement.split(' ', 1)[1]
            # path = dict_app.get(app)
            #
            # if path is None:
            #     speak('Application path not found')
            #     print('Application path not found')
            # else:
            #     speak('Launching: ' + app)
            #     os.system()

        elif 'send an email' in statement:
            speak('Who is this email addressed to? Please enter')
            recipient = input('Who is this email addressed to? Please enter')

            speak('What is the subject of the email? Please enter')
            subj = str(input('What is the subject of the email? Please enter'))

            speak('Now please, speak the message you want sent in 3, 2, 1')
            message = takecommand().lower()


            gmail_user = 'fiajan1602@gmail.com'
            gmail_password = 'Rasengan1602'

            sent_from = gmail_user
            to = recipient
            subject = subj
            body = message

            try:
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.ehlo()
                server.login(gmail_user, gmail_password)
                server.sendmail(sent_from, to, email_text)
                server.close()

                print
                'Email sent!'
            except:
                print
                'Something went wrong...'


        elif 'open 9anime' in statement:
            webbrowser.get('chrome').open_new("https://9anime.id")
            speak("9anime open now")
            time.sleep(5)

        elif 'time' in statement:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"the time is {strTime}")

        elif 'whats happening in the world' in statement or "whats the news" in statement:
            webbrowser.get('chrome').open_new("https://www.aljazeera.com/")
            speak('Here are the lastest news from the Al-Jazeera today, enjoy reading.')
            time.sleep(6)

        elif 'search' in statement:
            statement = statement.replace("search", "")
            webbrowser.open_new_tab(statement)
            time.sleep(5)

        elif 'i have a question' in statement:
            speak('I can answer to computational and geographical questions  and what question do you want to ask now')
            question = takeCommand()
            app_id = "TAX533-EKE3WG3KV6"
            client = wolframalpha.Client('R2K75H-7ELALHR35X')
            res = client.query(question)
            answer = next(res.results).text
            speak(answer)
            print(answer)

        elif 'who are you' in statement or 'what can you do' in statement:
            speak('I am Bukola version 1 point O your personal assistant. I am programmed to minor tasks like'
                  'opening youtube,google chrome, gmail and stackoverflow ,predict time,take a photo,search wikipedia,predict weather'
                  'In different cities, get top headline news from times of india and you can ask me computational or geographical questions too!')


        elif "who made you" in statement or "who created you" in statement or "who discovered you" in statement:
            speak("I was built by Filé Ajanaku")
            print("I was built by Filé")



        elif "how are you today" in statement or "how are you" in statement:
            speak('Im very well thank you and yourself?')
            print('Im very well thank you and yourself?')

            if response == 0:
                continue

            elif "I'm good thanks" in response or "not too bad, thank you" in response:
                speak("Glad to hear it Filé")
                print("Glad to hear it Filé")

            elif "not so good" in response or "not good" in response:
                speak("charge it")
                print("Charge it")





