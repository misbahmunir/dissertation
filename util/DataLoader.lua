require 'torch'
require 'image'

require 'ffmpeg'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(kwargs)
  self.splits = {
    train = {},
    test = {},
    val = {}
  }

  self.opt = {}

  self.splits.train.list = utils.getKwarg(kwargs, 'trainList')
  self.splits.test.list = utils.getKwarg(kwargs, 'testList')
  self.splits.val.list = utils.getKwarg(kwargs, 'valList')
  self.opt.dumpPath = utils.getKwarg(kwargs, 'dumpPath')
  self.opt.dumpFrames = utils.getKwarg(kwargs, 'dumpFrames')
  self.opt.seqLength = utils.getKwarg(kwargs, 'seqLength')
  self.opt.imageType = utils.getKwarg(kwargs, 'imageType')
  self.opt.videoHeight = utils.getKwarg(kwargs, 'videoHeight')
  self.opt.videoWidth = utils.getKwarg(kwargs, 'videoWidth')
  self.opt.scaledHeight = utils.getKwarg(kwargs, 'scaledHeight')
  self.opt.scaledWidth = utils.getKwarg(kwargs, 'scaledWidth')
  self.opt.maxClipLength = utils.getKwarg(kwargs, 'maxClipLength')
  self.opt.numChannels = utils.getKwarg(kwargs, 'numChannels')
  self.opt.desiredFPS = utils.getKwarg(kwargs, 'desiredFPS')
  self.opt.batchSize = utils.getKwarg(kwargs, 'batchSize')
  self.opt.cuda = utils.getKwarg(kwargs, 'cuda')

  local loadImagesOpt = {
    seqLength = self.opt.seqLength,
    imageType = self.opt.imageType,
    videoHeight = self.opt.videoHeight,
    videoWidth = self.opt.videoWidth,
    scaledHeight = self.opt.scaledHeight,
    scaledWidth = self.opt.scaledWidth,
    maxClipLength = self.opt.maxClipLength,
    numChannels = self.opt.numChannels,
    desiredFPS = self.opt.desiredFPS
  }

  for split, _ in pairs(self.splits) do
    self.splits[split].index = 1
    self.splits[split].file = paths.basename(self.splits[split].list)
    self.splits[split].paths, self.splits[split].labels = loadList(self.splits[split].list)
    self.splits[split].count = #self.splits[split].paths
    self.splits[split].dumpPath = paths.concat(self.opt.dumpPath, self.splits[split].file .. '_videos')
    if self.opt.dumpFrames == 1 then
      dumpVideoFrames(self.splits[split].paths, self.splits[split].dumpPath, loadImagesOpt)
    end
  end
  self.splits.train.shuffle = torch.randperm(self.splits.train.count)
  utils.printTime("Getting training data mean frame")
  self.splits.train.mean = getMeanTrainingImage(self.splits.train.paths, self.splits.train.dumpPath, loadImagesOpt)
end

function DataLoader:nextBatch(split)
	assert(split == 'train' or split == 'test' or split == 'val')

	local videoData = {}
  local frameLabels = {}

  while self.splits[split].index <= self.splits[split].count and #videoData < self.opt.batchSize do
  	local index
  	if split == 'train' then
  		index = self.splits[split].shuffle[self.splits[split].index]
		else
			index = self.splits[split].index
		end
  	local videoPath = self.splits[split].paths[index]
  	local videoLabel = self.splits[split].labels[index]

  	-- get video file
  	local videoFilename = paths.basename(videoPath)

  	-- get path of dumped frames for video
    videoPath = paths.concat(self.opt.dumpPath, self.splits[split].file .. '_videos', videoFilename .. '_frames')

    -- check if this video qualified to be read (extracted self.opt.seqLength or more frames)
    if paths.dirp(videoPath) then
      local framePath = paths.concat(videoPath, 'frame%d.' .. self.opt.imageType)
      local videoTensor = torch.Tensor(self.opt.seqLength, self.opt.numChannels, self.opt.scaledHeight, self.opt.scaledWidth)
      for i = 1, self.opt.seqLength do
        local frame = image.load(framePath % i, self.opt.numChannels, 'double')
        image.scale(videoTensor[i], frame) -- image.load reads in channels x height x width
        videoTensor[i] = videoTensor[i] - self.splits.train.mean
      end
      table.insert(videoData, videoTensor)
      table.insert(frameLabels, torch.Tensor(self.opt.seqLength):fill(videoLabel))
		end
		self.splits[split].index = self.splits[split].index + 1
  end

  if #videoData > 0 then
		local batch = {
		  data = torch.cat(videoData, 1):type('torch.FloatTensor'),
		  labels = torch.cat(frameLabels, 1):type('torch.FloatTensor'),
		}

		setmetatable(batch,
		  {__index = function(t, k)
		                  return {t.data[k], t.labels[k]}
		              end}
		);

		function batch:size()
		  return self.data:size(1)
		end

		return batch
  else
  	-- reset counters
  	self.splits[split].index = 1
  	return nil
  end
end

--[[
  Inputs:
    - videoListPath: path to a file containing paths to videos followed by label

  Adds each video in the file at videoListPath into a table. An example video
  list entry would be

  <videoPath> <label number>

  where videoPath looks like

  <intermediate directory>/<videoFilename>

  and is accessible from the working directory. Each video path and label is
  extracted and stored into separate tables. Both tables are returned.
]]--
function loadList(videoListPath)
	local videoPaths = {}
  local videoLabels = {}
  local file, err = io.open(videoListPath, 'rb')
  if err then
    utils.printTime(err)
    return
  else
    while true do
      local line = file:read()
      if line == nil then
        break
      end

      -- get tokens from line containing video path and label
      local tokens = {}
      for token in string.gmatch(line, "([^%s]+)") do
        table.insert(tokens, token)
      end
      local videoPath, videoLabel = unpack(tokens)

      table.insert(videoPaths, videoPath)
      table.insert(videoLabels, tonumber(videoLabel))
    end
  end

  return videoPaths, videoLabels
end

--[[
  Inputs:
    - videoPaths: table of paths to video files
    - fullDumpPath: directory to dump video frames
    - opt: parameters needed for video frame extraction

  Dumps each video in videoPaths to the fullDumpPath where fullDumpPath is

  <dumpPath>/<videoListFilename>_videos
  example: /data/train.txt_videos

  Each video path is extracted, as well as the actual video file name. A path in
  videoPaths would be

  <intermediate path>/<videoFilename>
  example: /data/videos/video1.avi

  The video is then read in from that path and opt.seqLength random frames of
  native  resolution are saved in chronological order at

  <fullDumpPath>/<videoFilename>_frames/frame<#>.<imageType>
  example: /data/train.txt_videos/video1.avi_frames/frame1.jpg

  TODO: Parallelize this.
]]--
function dumpVideoFrames(videoPaths, fullDumpPath, opt)
  -- make fullDumpPath
  local success, err = paths.mkdir(fullDumpPath)

  local numDumped = 0
  local numOmitted = 0

  for _, videoPath in pairs(videoPaths) do
    -- get video file name only
    local videoFilename = paths.basename(videoPath)

    local fullVideoPath = paths.concat('/data/UCF-101', videoPath)

    if paths.filep(fullVideoPath) then
      local video = ffmpeg.Video{
        path=fullVideoPath,
        width=opt.videoWidth,
        height=opt.videoHeight,
        fps=opt.desiredFPS,
        length=opt.maxClipLength,
        silent=true
      }

      local videoTensor = video:totensor({}) -- channels x height x width
      local numFrames = videoTensor:size()[1]
      if numFrames >= opt.seqLength then
        -- segment video frames into semi equally sized segments
        -- algorithm: http://stackoverflow.com/a/7788204
        -- TODO: make the +1's random rather than front loaded
        local segmentSize = math.floor(numFrames / opt.seqLength)
        local extra = numFrames % opt.seqLength
        local normal = opt.seqLength - extra

        local frameRange = torch.range(1, numFrames)
        local segments = {}
        local startIndex = 1
        for i = 1, extra do
          local endIndex = startIndex + segmentSize
          table.insert(segments, torch.totable(frameRange[{ {startIndex, endIndex} }]))
          startIndex = endIndex + 1
        end

        for i = 1, normal do
          local endIndex = startIndex + segmentSize - 1
          table.insert(segments, torch.totable(frameRange[{ {startIndex, endIndex} }]))
          startIndex = endIndex + 1
        end

        -- select random frame from each segment
        local frameIndices = {}
        for i = 1, opt.seqLength do
          local segment = segments[i]
          table.insert(frameIndices, segment[torch.random(1, #segment)])
        end

        videoOutputDirectory = paths.concat(fullDumpPath, videoFilename .. '_frames')
        local success, err = paths.mkdir(videoOutputDirectory)
        if err then
          utils.printTime(err .. " for %s" % videoOutputDirectory)
        end

        for k, v in pairs(frameIndices) do
          image.save(paths.concat(videoOutputDirectory, 'frame%d.%s' % {k, opt.imageType}), videoTensor[v])
        end
        numDumped = numDumped + 1
      else
        numOmitted = numOmitted + 1
        utils.printTime("Did not dump '%s', on line %d, because it did not generate enough frames." % {videoPath, numDumped + numOmitted})
      end
    end
  end
  utils.printTime("Dumped %d videos into %s and omitted %d videos. Total = %d." % {numDumped, fullDumpPath, numOmitted, numDumped + numOmitted})
end


--[[
  Inputs:
    - videoPaths: table of paths to video files
    - fullDumpPath: directory to dump video frames
    - opt: parameters needed for video frame extraction

  Dumps each video in videoPaths to the fullDumpPath where fullDumpPath is

  <dumpPath>/<videoListFilename>_videos
  example: /data/train.txt_videos

  Each video path is extracted, as well as the actual video file name. A path in
  videoPaths would be

  <intermediate path>/<videoFilename>
  example: /data/videos/video1.avi

  The video is then read in from that path and opt.seqLength random frames of
  native  resolution are saved in chronological order at

  <fullDumpPath>/<videoFilename>_frames/frame<#>.<imageType>
  example: /data/train.txt_videos/video1.avi_frames/frame1.jpg

  TODO: Parallelize this.
]]--

--[[
  Inputs:
    - videoPaths: table of paths to video files
    - fullDumpPath: directory to dump video frames
    - opt: parameters needed for video frame extraction

  Get the mean video frame after scaling. The mean frame is calculated in an
  unorthodox way because of possible oveflow and memory errors.
]]--
function getMeanTrainingImage(videoPaths, fullDumpPath, opt)
  local means = {0, 0, 0}
  local numFrames = 0

  for _, videoPath in pairs(videoPaths) do
    -- get video file
    local videoFilename = paths.basename(videoPath)

    -- get path of dumped frames for video
    videoPath = paths.concat(fullDumpPath, videoFilename .. '_frames')

    -- check if this video qualified to be read (had opt.seqLength or more frames)
    if paths.dirp(videoPath) then
      local framePath = paths.concat(videoPath, 'frame%d.' .. opt.imageType)
      for i = 1, opt.seqLength do
        numFrames = numFrames + 1
        local frame = image.load(framePath % i, opt.numChannels, 'double')
        frame = image.scale(frame, '%dx%d' % {opt.scaledWidth, opt.scaledHeight}) -- scales with string 'WxH', outputs channels x height x width
        for channel = 1, opt.numChannels do
          means[channel] = means[channel] + (frame[channel]:mean() - means[channel]) / numFrames
        end
      end
    end
  end

  local meanImage = torch.Tensor(opt.numChannels, opt.scaledHeight, opt.scaledWidth)
  for channel = 1, opt.numChannels do
    meanImage[channel]:fill(means[channel])
  end

  return meanImage
end
