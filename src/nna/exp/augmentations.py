import numpy as np
import torch
import librosa

#mix two of them by time
# time streching and

# run probabilities on the background
#


def randomMerge(data, count=1, splitIndex=-1):
    """
  Augment data by merging random samples.
  Increase data size to augmentad_size.
  Select required number of samples from data and merge from given splitIndex
  assumes [batch,sequence,...]
  does not modify data
  """
    if count < 1:
        return None

    sequence_Len = data.shape[1]
    sampleCount = data.shape[0]
    if splitIndex >= sequence_Len or splitIndex < -1:
        print("splitIndex should be less than sequence size")
        return None
    if sampleCount < 1:
        print("needs at least 2 samples")
        return None

    # Augment post samples
    augmentad = data[:]
    NewOnes = np.empty((0, *augmentad.shape[1:]))
    patience = 0
    while NewOnes.shape[0] < count:
        left = count - NewOnes.shape[0]
        left = sampleCount if left > sampleCount else left
        new = np.empty((left, *augmentad.shape[1:]))
        # randomly select 2*left # of samples

        first = augmentad[torch.randperm(augmentad.shape[0])[:left]].reshape(
            -1, *augmentad.shape[1:])
        second = augmentad[torch.randperm(augmentad.shape[0])[:left]].reshape(
            -1, *augmentad.shape[1:])
        if splitIndex == -1:
            middle = torch.randint(0, sequence_Len, (1,))[0]
        else:
            middle = splitIndex
        new[:, 0:middle], new[:, middle:] = first[:, 0:middle], second[:,
                                                                       middle:]
        NewOnes = np.concatenate([NewOnes, new])
        NewOnes = np.unique(NewOnes, axis=0)
        patience += 1
        if patience > (count * 10):
            print("samples not random enough")
            break
    return NewOnes


def randomConcat(data, y, count=1):
    """
  Augment data by concataneting random samples.
  Generate count number of them.
  assumes [sampleIndex,sequence,...]
  does not modify data
  return array of new data and Y
  """
    if count < 1:
        return None

    # sequence_Len = data.shape[1]
    sampleCount = data.shape[0]

    if sampleCount < 1:
        print("needs at least 2 samples")
        return None

    # Augment post samples
    augmentad = data[:]
    NewOnes = np.empty((0, augmentad.shape[1] * 2, *augmentad.shape[2:]),
                       dtype=augmentad.dtype)
    NewOnesY = np.empty((0, *y.shape[1:]), dtype=y.dtype)
    patience = 0
    while NewOnes.shape[0] < count:
        left = count - NewOnes.shape[0]
        left = sampleCount if left > sampleCount else left
        #         new=np.empty((left,augmentad.shape[2]*2,*augmentad.shape[2:]))
        # randomly select 2*left # of samples
        firstIndexes = torch.randperm(augmentad.shape[0])[:left]
        secondIndexes = torch.randperm(augmentad.shape[0])[:left]
        first = augmentad[firstIndexes].reshape(-1, *augmentad.shape[1:])
        second = augmentad[secondIndexes].reshape(-1, *augmentad.shape[1:])
        firstY = y[firstIndexes]
        secondY = y[secondIndexes]
        new = np.concatenate([first, second], axis=1)
        newY = firstY | secondY

        NewOnes = np.concatenate([NewOnes, new])
        NewOnesY = np.concatenate([NewOnesY, newY])
        NewOnes, NewOnesIndex = np.unique(NewOnes, axis=0, return_index=True)
        NewOnesY = NewOnesY[NewOnesIndex]

        patience += 1
        if patience > (count * 10):
            print("samples not random enough")
            break
    return NewOnes, NewOnesY


def randomAdd(data, y, count=1, unique=True):
    """
  Augment data by adding random samples.
  Generate count number of them.
  assumes [sampleIndex,sequence,...]
  does not modify data
  return array of new data and Y
  """
    if count < 1:
        return None

    # sequence_Len = data.shape[1]
    sampleCount = data.shape[0]

    if sampleCount < 1:
        print("needs at least 2 samples")
        return None

    # Augment post samples
    NewOnes = np.empty((count, data.shape[1], *data.shape[2:]),
                       dtype=data.dtype)
    NewOnesY = np.empty((count, *y.shape[1:]), dtype=y.dtype)
    patience = 0
    generatedCount = 0
    while generatedCount < count:
        left = count - generatedCount
        left = sampleCount if left > sampleCount else left
        # randomly select 2*left # of samples
        firstIndexes = torch.randperm(data.shape[0])[:left]
        secondIndexes = torch.randperm(data.shape[0])[:left]
        first = data[firstIndexes].reshape(-1, *data.shape[1:])
        second = data[secondIndexes].reshape(-1, *data.shape[1:])
        firstY = y[firstIndexes]
        secondY = y[secondIndexes]
        new = first * 0.5 + second * 0.5
        #         new=np.concatenate([first,second],axis=1)
        newY = np.logical_or(firstY, secondY).reshape(-1, *y.shape[1:])

        NewOnes[generatedCount:generatedCount + new.shape[0]] = new[:]
        NewOnesY[generatedCount:generatedCount + newY.shape[0]] = newY[:]

        if unique:
            NewOnes, NewOnesIndex = np.unique(NewOnes,
                                              axis=0,
                                              return_index=True)
            NewOnesY = NewOnesY[NewOnesIndex]
            patience += 1
            if patience > (count * 10):
                print("samples not random enough")
                break

        generatedCount += new.shape[0]

    return NewOnes, NewOnesY


def time_stretch(data, output_length, time_stretch_factor, singleElement=False):
    """

  """

    def strecth(y, output_length, time_stretch_factor):
        dataAug = librosa.effects.time_stretch(y.reshape(-1),
                                               time_stretch_factor)
        dataAug = dataAug.astype(data.dtype)

        if len(dataAug) > output_length:
            dataAug = dataAug[:output_length]
        else:
            dataAug = np.pad(dataAug, (0, max(0, output_length - len(dataAug))),
                             "constant")

        dataAug = dataAug.reshape(y.shape)
        return dataAug

    if singleElement:
        return strecth(data, output_length, time_stretch_factor)

    for index, _ in enumerate(data):
        dataAug = strecth(data, output_length, time_stretch_factor)
        if index == 0:
            dataAugAll = np.empty(data.shape, dtype=data.dtype)
        dataAugAll[index] = dataAug[:]

    return dataAugAll


def pitch_shift(data, sr, pitch_shift_n_steps, singleElement=False):
    """

  """

    def shift(y, sr, pitch_shift_n_steps):
        dataAug = librosa.effects.pitch_shift(y.reshape(-1), sr,
                                              pitch_shift_n_steps)
        dataAug = dataAug.astype(data.dtype)
        dataAug = dataAug.reshape(y.shape)
        return dataAug

    if singleElement:
        y = data
        return shift(y, sr, pitch_shift_n_steps)

    for index, y in enumerate(data):
        dataAug = shift(y, sr, pitch_shift_n_steps)
        if index == 0:
            dataAugAll = np.empty(data.shape, dtype=data.dtype)
        dataAugAll[index] = dataAug[:]

    return dataAugAll


def addNoise(data, noise_factor):

    noise = np.random.randn(*data.shape)
    noise = (noise_factor * noise).astype(data.dtype)
    augmented_data = data + noise
    # Cast back to same data type
    return augmented_data


#
# def linearAugmentation(data,AugIds,AugParams):
#     if len(AugIds)!=AugParams:
#         print("ERROR")
#         return None


class addNoiseClass(object):
    """
  """

    def __init__(self, noise_factor):
        self.noise_factor = noise_factor

    def __call__(self, sample):
        x, y = sample
        x = addNoise(x, self.noise_factor)
        return x, y


class pitch_shift_n_stepsClass(object):
    """
  """

    def __init__(self, sr, pitch_shift_n_steps):
        self.sr = sr
        self.pitch_shift_n_steps = pitch_shift_n_steps

    def __call__(self, sample):
        x, y = sample
        if isinstance(self.pitch_shift_n_steps, list):
            pitch_shift_n_steps = np.random.choice(self.pitch_shift_n_steps)
        else:
            pitch_shift_n_steps = self.pitch_shift_n_steps
        x = pitch_shift(x, self.sr, pitch_shift_n_steps, singleElement=True)
        return x, y


class time_stretchClass(object):
    """
  """

    def __init__(self, output_length, time_stretch_factor, isRandom=False):
        self.time_stretch_factor = time_stretch_factor
        self.output_length = output_length
        self.isRandom = isRandom

    def __call__(self, sample):
        x, y = sample
        if self.isRandom:
            assert isinstance(self.time_stretch_factor, list)
            lower, upper = self.time_stretch_factor
            time_stretch_factor_val = (upper - lower) * \
                np.random.random_sample() + lower
        else:
            time_stretch_factor_val = self.time_stretch_factor
        x = time_stretch(x,
                         self.output_length,
                         time_stretch_factor_val,
                         singleElement=True)
        return x, y


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, maxMelLen, sampling_rate):
        # sr = 44100 etc
        self.maxMelLen = maxMelLen
        self.sampling_rate = sampling_rate

    def __call__(self, sample):
        x, y = sample

        mel = librosa.feature.melspectrogram(y=x.reshape(-1),
                                             sr=self.sampling_rate)
        an_x = librosa.power_to_db(mel, ref=np.max)
        an_x = an_x.astype("float32")
        y = y.astype('float32')
        an_x = an_x[:, :self.maxMelLen]
        x = an_x.reshape(1, *an_x.shape[:])

        return torch.from_numpy(x), torch.from_numpy(np.asarray(y))



class shiftClass(object):
    """
  """

    def __init__(self, roll_rate, isRandom=False):
        self.roll_rate = roll_rate
        self.isRandom = isRandom

    def __call__(self, sample):
        x, y = sample

        shift = int(x.size * self.roll_rate)
        if self.isRandom:
            shift = np.random.randint(0, shift)
        x = np.roll(x, shift)

        return x, y


def createJamFiles():
    import jams

    from pathlib import Path

    import exp.runutils
    jamFolder = Path("/scratch/enis/data/nna/labeling/splitsJams")
    sourcePath = Path("/scratch/enis/data/nna/labeling/splits")

    humanresults = exp.runutils.loadLabels(labelsbyhumanpath)  #  pylint: disable=E0602
    for f in humanresults:
        # f="NIGLIQ2_20160702_002037_1368m_33s__1368m_43s.mp3"
        f = Path(f)
        fileStem = f.stem

        # Load the audio file
        infile = sourcePath / f
        y, sr = librosa.load(str(infile))

        # Compute the track duration
        track_duration = librosa.get_duration(y=y, sr=sr)

        tags4file = humanresults[str(f)]

        # Construct a new JAMS object and annotation records
        jam = jams.JAMS()

        # Store the track duration
        jam.file_metadata.duration = track_duration

        beat_a = jams.Annotation(namespace="tag_open")
        beat_a.annotation_metadata = jams.AnnotationMetadata(
            data_source="me@enisberk.com")

        for aTag in tags4file:
            beat_a.append(time=0, duration=track_duration, value=aTag)

        # Store the new annotation in the jam
        jam.annotations.append(beat_a)

        # Save to disk
        jam.save(str(jamFolder / (fileStem + ".jams")))
