from time import strftime, gmtime, time

def timeit(func, name=None):
	start_time = gmtime()
	start = time()
	result = func()
	end = time()
	end_time = gmtime()
	print('{}: Start "{}", End "{}", Duration "{:.2f}s"'.format(name if name else func.__name__, 
													 strftime("%d %b %H:%M:%S", start_time), 
													 strftime("%d %b %H:%M:%S", end_time), 
													 end - start))

	return result