def plot_embedding(resultq, resultg, query_id, gallery_id, title):
	# print('resultq',resultq.shape)
	data = np.concatenate((resultq, resultg),axis=0)
	# print('data',data.shape)

	x_min, x_max = np.min(data, 0), np.max(data, 0)
	x_min = x_min - 20
	x_max = x_max + 20    
	resultq = (resultq - x_min) / (x_max - x_min)
	resultg = (resultg - x_min) / (x_max - x_min)
	# plt.cm.Set1(i)
	fig = plt.figure()
	ax = plt.subplot(111)
	qid=list(set(query_id))
	# print('qid',qid)

	gid=list(set(gallery_id))
	# xid = random.sample(gid,10) 
	# print(xid)
	# xid = [102, 85, 266, 312, 192, 45, 257, 232, 41, 108]
	xid = [85, 257,266]
	# print('gid',gid)
	
	# cmap = get_cmap(len(qid))
	# cmap1 = get_cmap(len(gid))
	color=['r','orange','gray','yellow','g','cyan','b','violet','pink','k']
	for i in range(resultq.shape[0]):
		# print('query_id',query_id[i] )
		# print('resultq[i, 0]', resultq[i, 0])
		# print(qid.index(query_id[i]))
		if query_id[i] in xid:   
			plt.text(resultq[i, 0], resultq[i, 1], '*',
			 color=color[xid.index(query_id[i])] ,
			 fontdict={'weight': 'bold', 'size':9})
	for i in range(resultg.shape[0]):
		# print('gallery_id',gallery_id[i] )
		if gallery_id[i] in xid:
			plt.text(resultg[i, 0],resultg[i, 1], '.',
			 color=color[xid.index(gallery_id[i])],
			 fontdict={'weight': 'bold', 'size':12})
	plt.xticks([])
	plt.yticks([])
	#plt.title(title)
	return fig