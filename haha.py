import urllib
import urllib3

URL_IP = 'http://httpbin.org/ip'

def use_simple_urllib2():
    rep = urllib.request.urlopen(URL_IP)
    print (">>>Response Headers: ")
    print (rep)
    print ('>>>Response Body')
    #print ''.join([line for line in rep])

if __name__ == '__main__':
    print ('>>>Use simple urllib2:')
    use_simple_urllib2()