# ==============================CS-199==================================
# FILE:			MyAI.py
#
# AUTHOR: 		Justin Chung
#
# DESCRIPTION:	This file contains the MyAI class. You will implement your
#				agent in this file. You will write the 'getAction' function,
#				the constructor, and any additional helper functions.
#
# NOTES: 		- MyAI inherits from the abstract AI class in AI.py.
#
#				- DO NOT MAKE CHANGES TO THIS FILE.
# ==============================CS-199==================================

from AI import AI
from Action import Action

import numpy as np
import math

class MyAI( AI ):

	def __init__(self, rowDimension, colDimension, totalMines, startX, startY):
		self.board = Board(rowDimension, colDimension)
		self.totalTime = 0.0
		self.totalMines = totalMines

		self.prevX = startX
		self.prevY = startY

		self.frontier = {(startX, startY)}
		self.flagFrontier = set()
		self.nonzeroSet = set()

	def getAction(self, number: int) -> "Action Object":
		"""Return the current action this turn. Number is the label of the previous turn."""

		purgeSet = set()

		if (self.board.getBoard()[self.prevX][self.prevY] == self.board.COVERED):
			self.board.setBoard(self.prevX, self.prevY, number)
			purgeSet = self.board.getPurgableTiles(self.prevX, self.prevY)
			self.nonzeroSet = {tile for tile in self.nonzeroSet if (tile[0], tile[1]) not in purgeSet}

		# Add to nonzero set if last action had a nonzero label
		if number > 0 and (self.prevX, self.prevY) not in purgeSet:
			self.nonzeroSet.add((self.prevX, self.prevY, number))

		# Update frontier with all nonzero tiles and current tile
		self.frontier.update(self.board.getUncoverableTiles(self.nonzeroSet | {(self.prevX, self.prevY, number)}))

		# Update flag frontier with any flaggable tiles from the nonzero set
		# NOTE: We don't need to check current tile, it will be a zero label if not already in nonzero set
		self.flagFrontier.update(self.board.getFlaggableTiles(self.nonzeroSet))

		# OPTIMIZE REPEATED CONDITIONS
		# Run logic if all frontiers are empty and more bombs still exist
		if len(self.flagFrontier) == 0 and len(self.frontier) == 0 and self.board.getCurMines() < self.totalMines:
			uncoverableTiles, flaggableTiles = self.board.getSentenceTiles(self.nonzeroSet)

			self.frontier.update(*uncoverableTiles)
			self.flagFrontier.update(*flaggableTiles)

			for x, y in self.flagFrontier:
				self.board.setBoard(x, y, Board.FLAGGED)

			# Guess because logic found no new tiles
			if len(self.flagFrontier) == 0 and len(self.frontier) == 0 and self.board.getCurMines() < self.totalMines:
				self.frontier.update(self.board.getRandomNeighbor(self.nonzeroSet))

		# END OPTIMIZATION

		# Exhausts flag frontier before uncovering anymore tiles
		if self.flagFrontier:
			x, y = self.flagFrontier.pop()
			purgeSet = self.board.getPurgableTiles(x, y)
			self.nonzeroSet = {tile for tile in self.nonzeroSet if (tile[0], tile[1]) not in purgeSet}
			return Action(AI.Action.FLAG, x, y)
		
		# Chooses next move from the frontier
		if len(self.frontier) > 0:
			self.prevX, self.prevY = self.frontier.pop()
			return Action(AI.Action.UNCOVER, self.prevX, self.prevY)

		return Action(AI.Action.LEAVE)
	
	def _printSet(self, a, title) -> None:
		"""DEBUG TOOL, does not work on labeled sets"""
		temp = set(map(lambda x: (x[0] + 1, x[1] + 1) , a))
		print(title + ": " + str(temp))

class Board:
	COVERED = -1
	FLAGGED = -2

	def __init__(self, rowDimension, colDimension) -> None:
		self.board = np.array([[self.COVERED for _ in range(rowDimension)] for _ in range(colDimension)])
		self.curMines = 0
		self.columns = colDimension
		self.rows = rowDimension
	
	def setBoard(self, x, y, label) -> None:
		self.board[x][y] = label

	def getBoard(self) -> np.ndarray:
		return self.board
	
	def getCurMines(self) -> int:
		return self.curMines

	def getUncoverableTiles(self, checkSet) -> set:
		"""Return set of all uncoverable tiles from a given set."""
		return {tile for x, y, label in checkSet for tile in self._getUncoverableTile(x, y, label)}

	def getFlaggableTiles(self, checkSet) -> set:
		"""Return set of all flaggable tiles from a given set. Updates board with flags."""
		# NOTE: Can possibly flag from getAction directly because of misleading function name.

		final = set()
		for x, y, label in checkSet:
			coveredNeighbors = self._getCoveredNeighbors(x, y)
			if len(coveredNeighbors) > 0 and self._getEffectiveLabel(x, y, label) == len(coveredNeighbors):
				# Flag all covered neighbors of the list with an effective label equal to its length
				while len(coveredNeighbors) > 0:
					x, y = coveredNeighbors.pop()
					final.add((x, y))
					self.board[x][y] = self.FLAGGED
					self.curMines += 1

		return final

	def getPurgableTiles(self, x, y) -> set:
		"""Return set of all purgable tiles from a given coordinate"""
		# Adds a neighbor of (x, y) if the number of neighbors x, y have is equal to the number of noncovered neighbors it has
		purgable = {(newX, newY) for newX, newY in self._getAllNeighbors(x, y) if len(self._getNoncoveredNeighbors(newX, newY)) == len(self._getAllNeighbors(newX, newY))}
		# Adds the given coordinate, (x, y), to the purgable set if the number of all its neighbors is equal to its number of noncovered neighbors
		if len(self._getNoncoveredNeighbors(x, y)) == len(self._getAllNeighbors(x, y)):
			purgable.add((x, y))
		return purgable

	def getSentenceTiles(self, checkSet) -> tuple:
		"""
		Returns tuples of new possible tiles from logic.
		0 index is list of uncoverable tiles.
		1 index is list of flaggable tiles.
		"""

		# Get all sentences and brute force compare to remove all overlapping subsets
		sentences = self._getSentences(checkSet)

		temp = {r - l for l in sentences for r in sentences if l.isProperSubset(r)}

		uncoverable = list()
		flaggable = list()

		# If possible mines are 0, the set must all be uncoverable
		# If possible mines equal set length, the set must all be flaggable
		for subset in temp:
			if subset.getMines() == 0:
				uncoverable.append(subset.getTiles())
			elif len(subset.getTiles()) == subset.getMines():
				flaggable.append(subset.getTiles())

		return (uncoverable, flaggable)

	def getRandomNeighbor(self, checkSet) -> set:
		"""Return random neighbor from a given set. If no neighbor is found, a random covered tile is returned."""
		# if checkSet is not empty, get all Sentences and find the tile with the least probable chance of having a mine
		if checkSet:
			sentences = self._getSentences(checkSet)

			probDict = dict()
			for sentence in sentences:
				numTiles = len(sentence.getTiles())
				numMines = sentence.getMines()
				sentenceProbability = numMines / numTiles
				for tile in sentence.getTiles():
					if tile not in probDict.keys():
						probDict[tile] = [0, 0]
					probDict[tile] = [probDict[tile][0] + sentenceProbability, probDict[tile][1] + 1]
			
			probList = []
			for key in probDict.keys():
				averageProbability = probDict[key][0] / probDict[key][1]
				probList.append((key, averageProbability))

			sortedProbs = sorted(probList, key=lambda x: x[1])
			returnedSet = set()
			returnedSet.add(sortedProbs[0][0])
			return returnedSet
			
		# Pure guessing
		for x in range(len(self.board)):
			for y in range(len(self.board[0])):
				if self.board[x][y] == self.COVERED:
					returnedSet = set()
					returnedSet.add((x, y))
					return returnedSet
		
		# Returns empty set if nothing is found
		return set()

	def _getSentences(self, checkSet) -> set:
		"""Return set of sentences from a given set."""
		sentences = set()
		for x, y, label in checkSet:
			neighbors = self._getCoveredNeighbors(x, y)
			sentences.add(Sentence(neighbors, self._getEffectiveLabel(x, y, label)))
		return sentences

	def _getUncoverableTile(self, x, y, label) -> set:
		"""Return set of all uncoverable tiles of a given tile."""
		return self._getCoveredNeighbors(x, y) if self._getEffectiveLabel(x, y, label) == 0 else set()

	def _getEffectiveLabel(self, x, y, label) -> int:
		"""EffectiveLabel(x, y) = Label(x, y) - NumMarkedNeighbors(x, y)"""
		return label - len(self._getFlaggedTiles(self._getNoncoveredNeighbors(x, y)))

	def _getPossibleNeighbors(self, x, y) -> list:
		"""Return list of eight possible coordinates around a coordinate"""
		return [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x, y-1), (x+1, y-1), (x+1, y), (x+1, y+1)]

	def _getNoncoveredNeighbors(self, x, y) -> set:
		"""Return set of uncovered and flagged neighbors around a coordinate"""
		final = self._getPossibleNeighbors(x, y)
		return set(filter(lambda pos : self._isValidNoncoveredNeighbor(pos[0], pos[1]), final))
	
	def _getCoveredNeighbors(self, x, y) -> set:
		"""Return set of all covered neighbors around a coordinate"""
		final = self._getPossibleNeighbors(x, y)
		return set(filter(lambda pos : self._isValidCoveredNeighbor(pos[0], pos[1]), final))
	
	def _getAllNeighbors(self, x, y) -> set:
		"""Return set of all neighbors around a coordinate"""
		final = self._getPossibleNeighbors(x, y)
		return set(filter(lambda pos : self._isValidNeighbor(pos[0], pos[1]), final))

	def _isValidCoveredNeighbor(self, x, y) -> bool:
		"""Return condition to check if coordinate exists on the board and covered."""
		return x >= 0 and x < self.columns and y >= 0 and y < self.rows and self.board[x][y] == self.COVERED
	
	def _isValidNoncoveredNeighbor(self, x, y) -> bool:
		"""Return condition to check if coordinate exists on the board and noncovered."""
		return x >= 0 and x < self.columns and y >= 0 and y < self.rows and (self.board[x][y] >= 0 or self.board[x][y] == self.FLAGGED)

	def _isValidNeighbor(self, x, y) -> bool:
		"""Return condition to check if coordinate exists on the board."""
		return x >= 0 and x < self.columns and y >= 0 and y < self.rows
	
	def _getFlaggedTiles(self, neighbors) -> set:
		"""Return set of all flagged tiles given a set of neighbors."""
		return set([(x, y) for x, y in neighbors if self.board[x][y] == self.FLAGGED])

	def _printSet(self, a, title) -> None:
		"""DEBUG TOOL"""
		temp = set(map(lambda x: (x[0] + 1, x[1] + 1) , a))
		print(title + ": " + str(temp))

class Sentence:

	def __init__(self, tiles, mines) -> None:
		self.tiles = tiles ## set of strings
		self.mines = mines ## int

	def __str__(self) -> str:
		return str(set(map(lambda x: (x[0] + 1, x[1] + 1), self.tiles))) + " = " + str(self.mines)

	def __sub__(self, other) -> bool:
		"""Return Sentence with overlapping tiles removed and mines subracted."""
		return Sentence(self.tiles - other.getTiles(), self.mines - other.getMines())
	
	def __eq__(self, other) -> bool:
		"""Returns true if two Sentence objects have the same neighbors and same number of mines"""
		if isinstance(other, self.__class__):
			if self.tiles == other.tiles and self.mines == other.mines:
				return True
		return False
	
	def __hash__(self): 
		return hash((tuple(self.tiles), self.mines))

	def getTiles(self) -> set:
		return self.tiles
	
	def getMines(self) -> int:
		return self.mines

	def getLength(self) -> int:
		return len(self.tiles)

	def isProperSubset(self, other) -> bool:
		"""Return if this Sentence is a proper subset (subset and not equal to) the other Sentence"""
		return len(self.tiles) != len(other.getTiles()) and self.tiles.issubset(other.getTiles())