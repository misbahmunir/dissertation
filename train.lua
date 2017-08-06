require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'LSTM'

require 'LRCN'
require 'util.DataLoader'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-trainList', '') -- necessary
cmd:option('-valList', '') -- necessary
cmd:option('-testList', '') -- necessary
cmd:option('-numClasses', '') -- necessary
cmd:option('-dumpFrames', 1) -- fresh run assumes video frames should be dumped
cmd:option('-dumpPath', 'data')
cmd:option('-imageType', 'jpg')
cmd:option('-videoHeight', '') -- necessary
cmd:option('-videoWidth', '') -- necessary
cmd:option('-scaledHeight', '') -- uses native height if unprovided
cmd:option('-scaledWidth', '') -- uses native width if unprovided
cmd:option('-maxClipLength', 72) -- used to capture max length video
cmd:option('-numChannels', 3)
cmd:option('-desiredFPS', 5)
cmd:option('-batchSize', 2) -- batches of videos

-- Model options
cmd:option('-batchnorm', 1)
cmd:option('-dropout', 0.5)
cmd:option('-seqLength', 8)
cmd:option('-lstmHidden', 256)

-- Optimization options
cmd:option('-numEpochs', 30)
cmd:option('-learningRate', 1e-6)
cmd:option('-lrDecayFactor', 0.5)
cmd:option('-lrDecayEvery', 5)
cmd:option("-weightDecay", 2.5e-2, "L2 regularization")
cmd:option("-weightInitializationMethod", "kaiming", "heuristic, xavier, xavier_caffe, or none")

-- Output options
cmd:option('-printEvery', 1) -- Print the loss after every n epochs
cmd:option('-checkpointEvery', 3) -- Save model, print train acc
cmd:option('-checkpointName', 'checkpoints/checkpoint') -- Save model

-- Backend options
cmd:option('-cuda', 1)

local opt = cmd:parse(arg)

-- Torch cmd parses user input as strings so we need to convert number strings to numbers
for k, v in pairs(opt) do
  if tonumber(v) then
    opt[k] = tonumber(v)
  end
end

assert(opt.trainList ~= '', "Need a list of videos to train on, with the label separated by whitespace.")
assert(opt.testList ~= '', "Need a list of videos to test on, with the label separated by whitespace.")
assert(opt.numClasses ~= '', "Need the number of video classes.")
if opt.dumpFrames == 1 then
  assert(opt.videoHeight ~= '', "Video frames are to be dumped; need native height.")
  assert(opt.videoWidth ~= '', "Video frames are to be dumped; need native width.")
end

if opt.scaledHeight == '' then
  assert(opt.videoHeight > 8 and opt.videoHeight % 8 == 0, "Native video height must be divisible by 8. You need to enter a scaled height.")
  opt.scaledHeight = opt.videoHeight
else
  assert(opt.scaledHeight > 8 and opt.scaledHeight % 8 == 0, "Scaled frame height must be divisible by 8.")
end

if opt.scaledWidth == '' then
  assert(opt.videoWidth > 8 and opt.videoWidth % 8 == 0, "Native video width must be divisible by 8. You need to enter a scaled width.")
  opt.scaledWidth = opt.videoWidth
else
  assert(opt.scaledWidth > 8 and opt.scaledWidth % 8 == 0, "Scaled frame width must be divisible by 8.")
end

local allowableImageTypes = {
    ['jpg'] = true,
    ['png'] = true,
    ['ppm'] = true,
    ['pgm'] = true
  }
if not allowableImageTypes[opt.imageType] then
  opt.imageType = 'jpg'
end

-- Set up GPU
opt.dtype = 'torch.FloatTensor'
if opt.cuda == 1 then
  require 'cunn'
  opt.dtype = 'torch.CudaTensor'
end

-- Initialize DataLoader to receive batch data
utils.printTime("Initializing DataLoader")
local loader = DataLoader(opt)

-- Frames have been dumped, so we don't want to do so when we load this again in testing
opt.dumpFrames = 0

-- Initialize model and criterion
utils.printTime("Initializing LRCN")
local model = LRCN(opt):type(opt.dtype)
if opt.weightInitializationMethod ~= "none" then
  model = require("weight-init")(model, opt.weightInitializationMethod)
end
local criterion = nn.ClassNLLCriterion():type(opt.dtype)

--[[
  Input:
    - model: an LRCN

  Trains a fresh LRCN from end to end. Also uses the opt parameters declared above.
]]--
function train(model)
  utils.printTime("Starting training for %d epochs" % {opt.numEpochs})

  local trainLossHistory = {}
  local valLossHistory = {}
  local valLossHistoryEpochs = {}

  local config = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay
  }
  local params, gradParams = model:getParameters()

  for i = 1, opt.numEpochs do
    collectgarbage()

    local epochLoss = {}
    local videosProcessed = 0

    if i % opt.lrDecayEvery == 0 then
      local oldLearningRate = config.learningRate
      config = {
        learningRate = oldLearningRate * opt.lrDecayFactor,
        weightDecay = opt.weightDecay
      }
    end

    local batch = loader:nextBatch('train')

    while batch ~= nil do
      if opt.cuda == 1 then
        batch.data = batch.data:cuda()
        batch.labels = batch.labels:cuda()
      end

      videosProcessed = videosProcessed + (batch:size() / opt.seqLength)

      local function feval(x)
        collectgarbage()

        if x ~= params then
          params:copy(x)
        end

        gradParams:zero()

        local modelOut = model:forward(batch.data)
        local frameLoss = criterion:forward(modelOut, batch.labels)
        local gradOutputs = criterion:backward(modelOut, batch.labels)
        local gradModel = model:backward(batch.data, gradOutputs)

        return frameLoss, gradParams
      end

      local _, loss = optim.adam(feval, params, config)
      table.insert(epochLoss, loss[1])

      batch = loader:nextBatch('train')
    end

    local epochLoss = torch.mean(torch.Tensor(epochLoss))
    table.insert(trainLossHistory, epochLoss)

    -- Print the epoch loss
    if (opt.printEvery > 0 and i % opt.printEvery == 0) then
      utils.printTime("Epoch %d training loss: %f" % {i, epochLoss})
    end

    -- Save a checkpoint of the model, its opt parameters, the training loss history, and the validation loss history
    if (opt.checkpointEvery > 0 and i % opt.checkpointEvery == 0) or i == opt.numEpochs then
      local valLoss = test(model, 'val', 'loss')
      utils.printTime("Epoch %d validation loss: %f" % {i, valLoss})
      table.insert(valLossHistory, valLoss)
      table.insert(valLossHistoryEpochs, i)

      local checkpoint = {
        opt = opt,
        trainLossHistory = trainLossHistory,
        valLossHistory = valLossHistory
      }

      local filename
      if i == opt.numEpochs then
        filename = '%s_%s.t7' % {opt.checkpointName, 'final'}
      else
        filename = '%s_%d.t7' % {opt.checkpointName, i}
      end

      -- Make sure the output directory exists before we try to write it
      paths.mkdir(paths.dirname(filename))

      -- Cast model to float so it can be used on CPU
      model:float()
      checkpoint.model = model
      torch.save(filename, checkpoint)

      -- Cast model back so that it can continue to beu sed
      model:type(opt.dtype)
      params, gradParams = model:getParameters()
      utils.printTime("Saved checkpoint model and opt at %s" % filename)
      collectgarbage()
    end
  end

  utils.printTime("Finished training")
end

--[[
  Inputs:
    - model: an LRCN
    - split: 'train', 'val', or 'test'
    - task: 'recognition', 'detection', or 'loss'

  Performs either action recognition accuracy, action detection accuracy, or 
  loss for a split based on what task the user inputs.

  Action recognition is done by calculating the scores for each frame. The 
  score for a video is the max of the average of its sequence of frames.

  Action detection is done by calculating the scores for each frame and then 
  getting the max score for each frame.
]]--
function test(model, split, task)
  assert(task == 'recognition' or task == 'detection' or task == 'loss')
  collectgarbage()
  utils.printTime("Starting %s testing on the %s split" % {task, split})

  local evalData = {}
  if task == 'recognition' or task == 'detection' then
    evalData.predictedLabels = {} -- predicted video or frame labels
    evalData.trueLabels = {} -- true video or frame labels
  else
    evalData.loss = 0 -- sum of losses
    evalData.numBatches = 0 -- total number of frames
  end

  local batch = loader:nextBatch(split)

  while batch ~= nil do
    if opt.cuda == 1 then
      batch.data = batch.data:cuda()
      batch.labels = batch.labels:cuda()
    end

    if task == 'recognition' then
      local numData = batch:size() / checkpoint.opt.seqLength
      local scores = model:forward(batch.data)

      for i = 1, numData do
        local startIndex = (i - 1) * checkpoint.opt.seqLength + 1
        local endIndex = i * checkpoint.opt.seqLength
        local videoFrameScores = scores[{ {startIndex, endIndex}, {} }]
        local videoScore = torch.sum(videoFrameScores, 1) / checkpoint.opt.seqLength
        local maxScore, predictedLabel = torch.max(videoScore[1], 1)
        table.insert(evalData.predictedLabels, predictedLabel[1])
        table.insert(evalData.trueLabels, batch.labels[i])
      end
    elseif task == 'detection' then
      local numData = batch:size()
      local scores = model:forward(batch.data)

      for i = 1, numData do
        local videoFrameScores = scores[i]
        local _, predictedLabel = torch.max(videoFrameScores, 1)
        table.insert(evalData.predictedLabels, predictedLabel[1])
        table.insert(evalData.trueLabels, batch.labels[i])
      end
    else
      local numData = batch:size()
      local scores = model:forward(batch.data)

      evalData.loss = evalData.loss + criterion:forward(scores, batch.labels)
      evalData.numBatches = evalData.numBatches + 1
    end

    batch = loader:nextBatch(split)
  end

  if task == 'recognition' or task == 'detection' then
    evalData.predictedLabels = torch.Tensor(evalData.predictedLabels)
    evalData.trueLabels = torch.Tensor(evalData.trueLabels)
    return torch.sum(torch.eq(evalData.predictedLabels, evalData.trueLabels)) / evalData.predictedLabels:size()[1]
  else
    return evalData.loss / evalData.numBatches
  end
end

train(model)