require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'LSTM'

require 'util.DataLoader'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Options
cmd:option('-checkpoint', '')
cmd:option('-split', 'test')
cmd:option('-cuda', 1)

local opt = cmd:parse(arg)

assert(opt.checkpoint ~= '', "Need a trained network file to load.")

-- Set up GPU
opt.dtype = 'torch.FloatTensor'
if opt.cuda == 1 then
	require 'cunn'
  opt.dtype = 'torch.CudaTensor'
end

-- Initialize model and criterion
utils.printTime("Initializing model")
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:type(opt.dtype)
local criterion = nn.ClassNLLCriterion():type(opt.dtype)

-- Initialize DataLoader to receive batch data
utils.printTime("Initializing DataLoader")
local loader = DataLoader(checkpoint.opt)

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
  	evalData.numBatches = 0 -- number of batches run
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

local testDetectionAcc = test(model, 'test', 'detection')
utils.printTime("Action detection accuracy on the test set: %f" % testDetectionAcc)
local testRecognitionAcc = test(model, 'test', 'recognition')
utils.printTime("Action recognition accuracy on the test set: %f" % testRecognitionAcc)