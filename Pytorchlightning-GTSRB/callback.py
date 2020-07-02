import pytorch_lightning as pl

class PyLoggerCallback(pl.callbacks.Callback):
	def __init__(self, logger, **kwargs):
		super(PyLoggerCallback, self).__init__(**kwargs)
		self.logger = logger

	def on_epoch_start(self, trainer, pl_module):
		msg = 'Processing Epoch {}'.format(trainer.current_epoch+1)
		self.logger.info(msg)

	def on_epoch_end(self, trainer, pl_module):
		msg = 'Epoch {} - Train Loss: {:.6f}, Val Loss: {:.6f}, Val Accuracy: {:.6f}'.format(
			trainer.current_epoch+1,
			pl_module.stat['train']['loss'][-1],
			pl_module.stat['validation']['loss'][-1],
			pl_module.stat['validation']['accuracy'][-1]
		)
		self.logger.info(msg)

	def on_test_end(self, trainer, pl_module):
		msg = 'Test Loss: {:.6f}, Test Accuracy: {:.6f}'.format(
			pl_module.stat['test']['loss'][-1],
			pl_module.stat['test']['accuracy'][-1]
		)
		self.logger.info(msg)
