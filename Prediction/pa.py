from creme import linear_model
from creme import compose
from creme import compat
from creme import metrics
from creme import model_selection
from creme import optim
from creme import preprocessing
from creme import stream
from sklearn import datasets
import numpy as np 
import math
import sys
import pickle


def pred_frames(data, model, metric_X, metric_Y, frames, tile_manhattan_error, act_tiles, pred_tiles, count, width, height, nrow_tiles, ncol_tiles):
	x_pred, y_pred = 0,0

	for k in range(len(frames)):
		[inp_k, x_act, y_act] = data[frames[k]]
		if(k == 0):
			x_pred, y_pred = model.predict_one(inp_k, False, None, None)
		else:
			inp_k['VIEWPORT_x'] = x_pred
			inp_k['VIEWPORT_y'] = y_pred
			x_pred, y_pred = model.predict_one(inp_k, False, None, None)	

		shift = 0
		if(x_act > x_pred):
			if(abs(x_act - x_pred) > abs(x_act - (x_pred+width))):
				x_pred = x_pred+width
				shift = 1
		else:
			if(abs(x_act - x_pred) > abs(x_act - (x_pred-width))):
				x_pred = x_pred-width
				shift = 2

		metric_X = metric_X.update(x_act, x_pred)
		metric_Y = metric_Y.update(y_act, y_pred)

		if(shift == 0):
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int(x_pred * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)
		elif(shift == 1):
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int((x_pred - width) * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)
		else:
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int((x_pred + width) * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)

		actual_tile_row = actual_tile_row-nrow_tiles if(actual_tile_row >= nrow_tiles) else actual_tile_row
		actual_tile_col = actual_tile_col-ncol_tiles if(actual_tile_col >= ncol_tiles) else actual_tile_col
		actual_tile_row = actual_tile_row+nrow_tiles if actual_tile_row < 0 else actual_tile_row
		actual_tile_col = actual_tile_col+ncol_tiles if actual_tile_col < 0 else actual_tile_col

		######################################################
		# print("x: "+str(x_act))
		# print("x_pred: "+str(x_pred))
		# print("y: "+str(y_act))	
		# print("y_pred: "+str(y_pred))
		# print("("+str(actual_tile_row)+","+str(actual_tile_col)+"),("+str(pred_tile_row)+","+str(pred_tile_col)+")")
		######################################################
		
		act_tiles.append((actual_tile_row, actual_tile_col))
		pred_tiles.append((pred_tile_row, pred_tile_col))

		tile_col_dif = ncol_tiles

		if actual_tile_col < pred_tile_col:
			tile_col_dif = min(pred_tile_col - actual_tile_col, actual_tile_col + ncol_tiles - pred_tile_col)
		else:
			tile_col_dif = min(actual_tile_col - pred_tile_col, ncol_tiles + pred_tile_col - actual_tile_col)

		tile_manhattan_error += abs(actual_tile_row - pred_tile_row) + abs(tile_col_dif)
		count = count+1

	return metric_X, metric_Y, tile_manhattan_error, count, act_tiles, pred_tiles



def build_model(data, frame_nos, max_frame, tot_objects, width, height, nrow_tiles, ncol_tiles, fps, pred_nframe):
	model = linear_model.PARegressor(C=0.01, mode=2, eps=0.001, data=data, learning_rate=0.005, rho=0.99)
	metric_X = metrics.MAE()
	metric_Y = metrics.MAE()
	manhattan_error = []
	x_mae = []
	y_mae = []
	count=0

	i=0
	tile_manhattan_error=0
	act_tiles, pred_tiles = [],[]
	chunk_frames = []

	#Initial training of first 5 seconds
	while True:
		curr_frame=frame_nos[i]
		if curr_frame<5*fps:
			i=i+1
			[inp_i,x,y]=data[curr_frame]
			model = model.fit_one(inp_i,x,y)
		else:
			break

	# Predicting frames and update model
	while True:
		curr_frame = frame_nos[i]
		nframe = min(pred_nframe, max_frame - frame_nos[i])

		if(nframe <= 0):
			break

		frames = {i}
		for k in range(i+1, len(frame_nos)):
			if(frame_nos[k] < curr_frame + nframe):
				frames.add(k)
			else:
				i=k
				break
		if(i!=k):
			i=k

		if(i==(len(frame_nos)-1)):
			break

		frames = sorted(frames)
		chunk_frames.append(frames)

		metric_X, metric_Y, tile_manhattan_error, count, act_tiles, pred_tiles = pred_frames(data, model, metric_X, metric_Y, frames, tile_manhattan_error, act_tiles, pred_tiles, count, width, height, nrow_tiles, ncol_tiles)
		model = model.fit_n(frames)

		manhattan_error.append(tile_manhattan_error*1.0 / count)
		x_mae.append(metric_X.get())
		y_mae.append(metric_Y.get())

		print("Manhattan Tile Error: "+str(tile_manhattan_error*1.0 / count))
		print(metric_X, metric_Y)
		print("\n")

	return act_tiles, pred_tiles, chunk_frames, manhattan_error, x_mae, y_mae
