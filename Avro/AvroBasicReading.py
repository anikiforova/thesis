
from avro.datafile import DataFileReader
from avro.io import DatumReader

filename = "screenshots_part-00000.avro"
 
file = open(filename, 'rb')
datum_reader = DatumReader()
file_reader = DataFileReader(file, datum_reader)
 
print(file_reader .meta)

file_reader = DataFileReader(file, datum_reader)
for datum in file_reader:
	print(datum['uri'])
	break

file.close()
