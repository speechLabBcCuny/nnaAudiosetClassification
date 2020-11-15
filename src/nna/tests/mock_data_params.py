# 14/Nov/2020
# from pathlib import Path
# a = glob.glob("/tank/data/nna/real/*/*")
# a1 = [(Path(i).stem) for i in a]
# a2 = [(Path(i).parent.stem,(Path(i).stem)) for i in a]
EXISTING_REGION_LOCATIONID = [('anwr', '31'), ('anwr', '32'), ('anwr', '33'),
                              ('anwr', '34'), ('anwr', '35'), ('anwr', '36'),
                              ('anwr', '37'), ('anwr', '38'), ('anwr', '39'),
                              ('anwr', '40'), ('anwr', '41'), ('anwr', '42'),
                              ('anwr', '43'), ('anwr', '44'), ('anwr', '45'),
                              ('anwr', '46'), ('anwr', '47'), ('anwr', '48'),
                              ('anwr', '49'), ('anwr', '50'), ('dalton', '01'),
                              ('dalton', '02'), ('dalton', '03'),
                              ('dalton', '04'), ('dalton', '05'),
                              ('dalton', '06'), ('dalton', '07'),
                              ('dalton', '08'), ('dalton', '09'),
                              ('dalton', '10'), ('dempster', '11'),
                              ('dempster', '12'), ('dempster', '13'),
                              ('dempster', '14'), ('dempster', '16'),
                              ('dempster', '17'), ('dempster', '19'),
                              ('dempster', '20'), ('dempster', '21'),
                              ('dempster', '22'), ('dempster', '23'),
                              ('dempster', '24'), ('dempster', '25'),
                              ('ivvavik', 'AR01'), ('ivvavik', 'AR02'),
                              ('ivvavik', 'AR03'), ('ivvavik', 'AR04'),
                              ('ivvavik', 'AR05'), ('ivvavik', 'AR06'),
                              ('ivvavik', 'AR07'), ('ivvavik', 'AR08'),
                              ('ivvavik', 'AR09'), ('ivvavik', 'AR10'),
                              ('ivvavik', 'SINP01'), ('ivvavik', 'SINP02'),
                              ('ivvavik', 'SINP03'), ('ivvavik', 'SINP04'),
                              ('ivvavik', 'SINP05'), ('ivvavik', 'SINP06'),
                              ('ivvavik', 'SINP07'), ('ivvavik', 'SINP08'),
                              ('ivvavik', 'SINP09'), ('ivvavik', 'SINP10'),
                              ('prudhoe', '11'), ('prudhoe', '12'),
                              ('prudhoe', '13'), ('prudhoe', '14'),
                              ('prudhoe', '15'), ('prudhoe', '16'),
                              ('prudhoe', '17'), ('prudhoe', '18'),
                              ('prudhoe', '19'), ('prudhoe', '20'),
                              ('prudhoe', '21'), ('prudhoe', '22'),
                              ('prudhoe', '23'), ('prudhoe', '24'),
                              ('prudhoe', '25'), ('prudhoe', '26'),
                              ('prudhoe', '27'), ('prudhoe', '28'),
                              ('prudhoe', '29'), ('prudhoe', '30'),
                              ('stinchcomb', '01-Itkillik'),
                              ('stinchcomb', '02-Colville2'),
                              ('stinchcomb', '03-OceanPt'),
                              ('stinchcomb', '04-Colville4'),
                              ('stinchcomb', '05-Colville5'),
                              ('stinchcomb', '06-Umiruk'),
                              ('stinchcomb', '07-IceRd'),
                              ('stinchcomb', '08-CD3'),
                              ('stinchcomb', '09-USGS'),
                              ('stinchcomb', '10-Nigliq1'),
                              ('stinchcomb', '11-Nigliq2'),
                              ('stinchcomb', '12-Anaktuvuk'),
                              ('stinchcomb', '13-Shorty'),
                              ('stinchcomb', '14-Rocky'),
                              ('stinchcomb', '15-FishCreek1'),
                              ('stinchcomb', '16-FishCreek2'),
                              ('stinchcomb', '17-FishCreek3'),
                              ('stinchcomb', '18-FishCreek4'),
                              ('stinchcomb', '19-Itkillik2'),
                              ('stinchcomb', '20-Umiat')]
IGNORE_LOCATION_ID = ['excerpts', 'dups']

EXISTING_YEARS = ['2010', '2013', '2016', '2018', '2019']
IGNORE_YEARS = ['2010', '2013', '2016']

EXISTING_SUFFIX = ['.flac', '.mp3', '.FLAC', '.mp3']

# in seconds, (4.5 hours + extra 30 minutes)
EXISTING_LONGEST_FILE_LEN = 5 * 60 * 60
