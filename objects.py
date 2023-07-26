import numpy as np
import scipy.stats as scst

class CellularObject:
	def __init__(self, x, y, tx_power_dBm, phy_rate=8e6, bandwidth=2e7, signal_spread=2.2e7, packet_size=1.46e3, header_delay=1e-4):
		self._location = np.array([x, y])
		self._sinr = None
		self._tx_dBm = tx_power_dBm
		self._phy_rate = phy_rate
		self._bandwidth = bandwidth
		self._spread = signal_spread
		self._packet_size = packet_size
		self._header_delay = 1e-4
		self._sinr = 0

	def computeReceptionMetrics(self, sinr):
		"""Set internal state for reception according to SINR
		
		Arguments:
				sinr {float} -- the SINR at reception
		"""
		self._sinr = sinr
		rTime = self._header_delay + self._packet_size / self._phy_rate
		self._per = self.computePER(sinr)

		self._throughput = self._per * self._packet_size / rTime

	def computeBER(self, sinr):
		return scst.norm.sf(np.sqrt(1/3 * sinr * self._spread / self._phy_rate))

	def computePER(self, sinr, packetSize=None):	
		if packetSize is None:
			packetSize = self._packet_size
		return 1 - (1 - self.computeBER(sinr)) ** (8 * packetSize)