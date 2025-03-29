#!/bin/python3

# Some basic functions that I universally use in my projects

import string
import re
import json
import random
import os
from pathlib import Path
import datetime

def writef(name,data,method="w"): # Just to make writing files a bit easier without having to f=open and f.close every time
	if type(data) == list or type(data) == dict:
		data = json.dumps(data)
	if type(data) == bytes and 'b' not in method:
		data = data.decode('utf-8')
	Path(name).touch()
	f=open(name,method)
	f.write(data)
	f.close()

def readf(name,method="r"):
	f=open(name,method)
	data = f.read()
	f.close()
	return data

def randlower(length): # Random lowercase string
	letters = string.ascii_lowercase + '0123456789'
	return ''.join(random.choice(letters) for i in range(length))

def mass_replace(text, match):
	# Create a regular expression  from the dictionary keys
	regex = re.compile("(%s)" % "|".join(map(re.escape, match.keys())))

	# For each match, look-up corresponding value in dictionary
	return regex.sub(lambda mo: match[mo.string[mo.start():mo.end()]], text)

def convert_timezone(time,timezone):
	#current = readconfig('http.cfg')['timezone'] Always have current timezone as EDT
	diff = {
		'HST': -6,
		'HDT': -5,
		'AKDT': -4,
		'PDT': -3,
		'MST': -3,
		'MDT': -2,
		'CDT': -1,
		'EDT': 0
	}
	return time + datetime.timedelta(hours=diff[timezone])

def readlastlines(fname,lines): # Read the last lines of a file
	with open(fname, "rb") as file:
		file.seek(-2, os.SEEK_END)
		i = 0
		while i < lines:
			if file.read(1) == b'\n':
				i += 1
				try:
					file.seek(-2, os.SEEK_CUR)
				except: # Just in case we go back too much
					break
			else:
				try:
					file.seek(-2, os.SEEK_CUR)
				except: # Just in case we go back too much
					break
		file.seek(2,os.SEEK_CUR) # Trim the leftovers
		return str(file.read().decode()[:-1])

configOverride = {}

# So we can override our config files via command line args
def overrideConfig(option, override):
	configOverride[option] = override

def readconfig(config):
	filename = config + '.config'
	if not os.path.exists('./config/'+filename):
		raise BaseException('Configuration file not found: %s' % (filename))
	data = json.loads(readf('./config/'+filename))
	for override in configOverride:
		if override in data:
			data[override] = configOverride[override]
	return data
