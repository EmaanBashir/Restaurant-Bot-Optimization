from PIL import Image, ImageTk
import tkinter as tk
import random
import math
import numpy as np
import time
from algorithms.aStar import astar_with_avoid_blocks
from algorithms.dijkstra import dijkstra
from algorithms.dfs import dfs
from algorithms.bfs import bfs

# Dimensions
WINDOW_SIZE =  (1160, 760)
TABLE_SIZE = (100, 80)
KITCHEN_SIZE = (80, 50)
DOOR_SIZE = (10, 80)
CHARGER_SIZE = (70, 50)
BOT_SIZE = (40, 40)
MAP_SIZE = BOT_SIZE

# Turning Angle
ANGLE_RANGE = math.pi/8
START_ANGLE = math.pi/16

# Battery
BATTERY_CAPACITY = 5000
MIN_BATTERY = 1000

# Order
class Order():
    def __init__(self, name, table):
        self.name = name
        self.collected = False
        self.delivered = False
        self.table = table

    # Collection status
    def isCollected(self):
        return self.collected
    
    # Order collected
    def collect(self):
        self.collected = True
    
    # Delivery status
    def isDelivered(self):
        return self.delivered
    
    # Order delivered
    def deliver(self):
        self.delivered = True

    # Get the table that placed the order
    def getTable(self):
        return self.table

# Brain class for controlling the robot
class Brain():

    # Initialize attributes
    def __init__(self,botp):
        self.bot = botp
        self.time = 0
        self.trainingSet = []
        self.orderTransitionTime = 0
        self.maxTransitionTime = 20
        self.turningLeft = False
        self.turningRight = False

    # Robot Behavior
    def thinkAndAct(self, canvas, targetL, targetR, chargerL, chargerR, x, y, sl, sr,\
                    battery, collision, order):
        newX = None
        newY = None
        self.time += 1
        self.bot.setTowardsCharger(False)

        # If the robot has decided a path
        if self.bot.getPath():
            target = self.bot.getPath()[0]
            # Calculate distance to left and right sensor
            coordinates = tuple(x * BOT_SIZE[0] + BOT_SIZE[0]/2 for x in target)
            dl = self.bot.distanceToLeftSensor(coordinates[0], coordinates[1])
            dr = self.bot.distanceToRightSensor(coordinates[0], coordinates[1])
        # If robot does not have a path yet
        else:
            dl = 0
            dr = 0
            speedLeft = 0
            speedRight = 0
        
        # Adjust speed based on sensor values
        if dr > dl:
            speedLeft = 2.0
            speedRight = -2.0
 
        elif dr<dl:
            speedLeft = -2.0
            speedRight = 2.0           
         
        # Handle approximately equal values of sensors
        if abs(dr-dl)<dl*0.1: 
            if self.bot.getPath() and dl > self.bot.distanceToPoint(coordinates[0], coordinates[1]) and \
            dr > self.bot.distanceToPoint(coordinates[0], coordinates[1]) and self.bot.distanceToPoint(coordinates[0], coordinates[1]) > self.bot.getSize()[1]/2:
                speedLeft = 2.0
                speedRight = -2.0
            else:
                speedLeft = 5.0
                speedRight = 5.0

        # Remove reached target from the path
        if self.bot.getPath() and  dl + dr < BOT_SIZE[0]:  
            self.bot.getPath().remove(target)
        
        # Check if the robot target is the charger
        if targetL == chargerL:
            minTargetVal = 200
            self.bot.setTowardsCharger(True)
        else:
            minTargetVal = 100
        
        # Stop if robot is near charger
        if targetL+targetR>minTargetVal:
            speedLeft = 0.0
            speedRight = 0.0
            
            # Handle order collection and delivery
            if order:
                self.orderTransitionTime += 1
                if self.orderTransitionTime == self.maxTransitionTime:
                    self.orderTransitionTime = 0
                    self.bot.endPath()
                    if not order.isCollected():
                        order.collect()
                    elif not order.isDelivered():
                        order.deliver()     
        
        # Handle low battery
        if battery<MIN_BATTERY and not isinstance(self.bot.target, Table):
            targetL = chargerL
            targetR = chargerR
            minTargetVal = 200
            self.bot.setTowardsCharger(True)
            if chargerR>chargerL:
                speedLeft = 2.0
                speedRight = -2.0
            elif chargerR<chargerL:
                speedLeft = -2.0
                speedRight = 2.0
            if abs(chargerR-chargerL)<chargerL*0.1:
                speedLeft = 5.0
                speedRight = 5.0
        if chargerL+chargerR>200 and battery<BATTERY_CAPACITY-10:
            speedLeft = 0.0
            speedRight = 0.0

        # Handle collision
        collisionWithBot, collisionWithCustomer, collisionWithPassiveObject = collision
        # collision with customer
        if collisionWithCustomer[0]:
            speedLeft = 0
            speedRight = 0

        # Avoid going out of the screen area
        if x>WINDOW_SIZE[0] - BOT_SIZE[0]/2 + 2:
            newX = WINDOW_SIZE[0] - BOT_SIZE[0]/2
            if targetL > targetR:
                speedLeft = -5.0
                speedRight = 5.0
            else:
                speedLeft = 5.0
                speedRight = -5.0
        if x<BOT_SIZE[0]/2 - 2:
            newX = BOT_SIZE[0]/2
            if targetL > targetR:
                speedLeft = -5.0
                speedRight = 5.0
            else:
                speedLeft = 5.0
                speedRight = -5.0
        if y>WINDOW_SIZE[1] - BOT_SIZE[0]/2 + 2:
            newY = WINDOW_SIZE[1] - BOT_SIZE[0]/2
            if targetL > targetR:
                speedLeft = -5.0
                speedRight = 5.0
            else:
                speedLeft = 5.0
                speedRight = -5.0
        if y<BOT_SIZE[0]/2 - 2:
            newY = BOT_SIZE[0]/2
            if targetL > targetR:
                speedLeft = -5.0
                speedRight = 5.0
            else:
                speedLeft = 5.0
                speedRight = -5.0
    
        return speedLeft, speedRight, newX, newY

# Robot class
class Bot():

    def __init__(self,namep, x, y, canvasp, method, id):
        # Initialize bot attributes
        self.name = namep
        self.canvas = canvasp
        self.id = id
        self.x = x
        self.y = y
        self.theta = 1.5*math.pi
        self.ll = 60 
        self.sl = 0.0
        self.sr = 0.0
        self.battery = BATTERY_CAPACITY
        self.size = BOT_SIZE
        self.orders = []
        self.currentOrder = None
        self.towardsKitchen = False
        self.towardsTable = False
        self.towardsCharger = False
        self.target = None
        self.path = None
        # Initialize image attributes
        imgSize = (int(self.size[0]*0.7), int(self.size[1]*0.7))
        imgFile = Image.open("images/food.png")
        imgFile = imgFile.resize(imgSize, Image.LANCZOS)
        self.foodImage = ImageTk.PhotoImage(imgFile)

        # Select the path finding algorithm
        if method == 'astar':
            self.pathMethod = astar_with_avoid_blocks
        elif method == 'dijkstra':
            self.pathMethod = dijkstra
        elif method == 'dfs':
            self.pathMethod = dfs
        elif method == 'bfs':
            self.pathMethod = bfs

    # Decision making of the bot
    def thinkAndAct(self, agents, passiveObjects, canvas):
        if self.battery == BATTERY_CAPACITY and len(self.orders) == 0 and not self.currentOrder:
            return

        # Handle the state of the robot
        if self.currentOrder == None:
            if self.getNoOfOrders() > 0:
                self.currentOrder = self.orders[0]
                self.orders.remove(self.currentOrder)
                self.path = None
                self.towardsKitchen = True
        elif not self.currentOrder.isCollected():
            self.towardsKitchen = True
            self.towardsTable = False
        elif self.currentOrder.isCollected() and not self.currentOrder.isDelivered():
            self.towardsKitchen = False
            self.towardsTable = True
        elif self.currentOrder.isDelivered():
            self.towardsKitchen = False
            self.towardsTable = False
            if self.getNoOfOrders() > 0:
                self.currentOrder = self.orders[0]
                self.orders.remove(self.currentOrder)
                self.towardsKitchen = True

        # Select the target of the bot
        if self.towardsKitchen:
            targetL, targetR = self.senseKitchen(passiveObjects)
            self.target = self.getKitchen(passiveObjects)
        elif self.towardsTable:
            targetL, targetR = self.senseTable(self.currentOrder.getTable())
            self.target = self.currentOrder.getTable()
        else:
            targetL, targetR = self.senseChargers(passiveObjects)
            self.target = self.getCharger(passiveObjects)
       
       # Set target = Charger if batttery is low
        if self.battery<MIN_BATTERY and not self.towardsTable:
            targetL, targetR = self.senseChargers(passiveObjects)
            if self.target != self.getCharger(passiveObjects):
                self.target = self.getCharger(passiveObjects)
                self.endPath()

        # Find path if the bot does not already have a path
        if not self.path:
            
            startPoint = (int(math.floor(self.x/MAP_SIZE[0])), int(math.floor(self.y/MAP_SIZE[1])))
            endx, endy = self.target.getLocation()
            endPoint = (int(math.floor(endx/MAP_SIZE[0])), int(math.floor(endy/MAP_SIZE[1])))
          
            self.avoidBlocks = self.getAvoidBlocks(passiveObjects, self.target)
            self.path = self.pathMethod(startPoint, endPoint, self.avoidBlocks)
            
            # Draw the path
            if self.path:
                canvas.delete("pathLine" + str(self.id))
                for i in range(len(self.path)-1):
                    canvas.create_line(self.path[i][0] * MAP_SIZE[0] + MAP_SIZE[0]/2, self.path[i][1] * MAP_SIZE[1] + MAP_SIZE[1]/2, self.path[i+1][0] * MAP_SIZE[0] + MAP_SIZE[0]/2, self.path[i+1][1] * MAP_SIZE[1] + MAP_SIZE[1]/2, fill = 'red', width = 2, tags = "pathLine" + str(self.id))
            
            # # Mark the blocks to be avoided
            # for i in range (len(self.avoidBlocks)-1):
            #     canvas.create_rectangle(self.avoidBlocks[i][0]* MAP_SIZE[0] + MAP_SIZE[0]/2, self.avoidBlocks[i][1]* MAP_SIZE[1] + MAP_SIZE[1]/2, self.avoidBlocks[i][0]* MAP_SIZE[0] + MAP_SIZE[0]/2 + 3, self.avoidBlocks[i][1]* MAP_SIZE[0] + MAP_SIZE[0]/2 + 3, fill = "red")
        
        # Sense the charger
        chargerL, chargerR = self.senseChargers(passiveObjects)
        
        # Get collision status
        collision = [self.collisionWithBot(agents), self.collisionWithCustomer(agents), self.collisionWithPassiveObject(passiveObjects)]
        
        # Handle collision with another bot
        if collision[0][0]:
            startPoint = (int(math.floor(self.x/MAP_SIZE[0])), int(math.floor(self.y/MAP_SIZE[1])))
            endx, endy = self.target.getLocation()
            endPoint = (int(math.floor(endx/MAP_SIZE[0])), int(math.floor(endy/MAP_SIZE[1])))

            # Get the cell to avoid in order to avoid the colliding bot
            avoidBot = self.getAvoidBot(passiveObjects, collision[0][1], self.target)
            # Find new path
            self.path = self.pathMethod(startPoint, endPoint, avoidBot)
            # Draw the path
            if self.path:
                canvas.delete("pathLine" + str(self.id))
                for i in range(len(self.path)-1):
                    canvas.create_line(self.path[i][0] * MAP_SIZE[0] + MAP_SIZE[0]/2, self.path[i][1] * MAP_SIZE[1] + MAP_SIZE[1]/2, self.path[i+1][0] * MAP_SIZE[0] + MAP_SIZE[0]/2, self.path[i+1][1] * MAP_SIZE[1] + MAP_SIZE[1]/2, fill = 'blue', width = 2, tags = "pathLine" + str(self.id))

        # think and act through brain
        self.sl, self.sr, xx, yy = self.brain.thinkAndAct\
            (canvas, targetL, targetR, chargerL, chargerR, self.x, self.y,
             self.sl, self.sr,self.battery, collision, self.currentOrder)

        if xx != None:
            self.x = xx
        if yy != None:
            self.y = yy
        
    # Set brain of the bot
    def setBrain(self,brainp):
        self.brain = brainp

    # Get the blocks to avoid due to passive objects on the map
    def getAvoidBlocks(self, passiveObjects, target = None):
        avoidblocks = []
        for p in passiveObjects:
            if p != target and not isinstance(p, Door):
                cx, cy = p.getLocation()
                w, h = p.getSize()
                xRange = (cx - w/2 - MAP_SIZE[0]/2, cx + w/2 + MAP_SIZE[0]/2)
                yRange = (cy - h/2 - MAP_SIZE[1]/2, cy + h/2 + MAP_SIZE[1]/2)
                for i in range(int(xRange[0]), int(xRange[1])):
                    for j in range(int(yRange[0]), int(yRange[1])):
                        xx = int(math.floor(i/MAP_SIZE[0]))
                        yy = int(math.floor(j/MAP_SIZE[1]))
                        avoidblocks.append((xx, yy))
        return list(set(avoidblocks))
    
    # Get the blocks to avoid due to a colliding bot and passive objects on the map
    def getAvoidBot(self, passiveObjects, bot, target = None):
        avoidblocks = []

        # Avoid passive objects
        for p in passiveObjects:
            if p != target and not isinstance(p, Door):
                cx, cy = p.getLocation()
                w, h = p.getSize()
                xRange = (cx - w/2 - MAP_SIZE[0]/2, cx + w/2 + MAP_SIZE[0]/2)
                yRange = (cy - h/2 - MAP_SIZE[1]/2, cy + h/2 + MAP_SIZE[1]/2)
                for i in range(int(xRange[0]), int(xRange[1])):
                    for j in range(int(yRange[0]), int(yRange[1])):
                        xx = int(math.floor(i/MAP_SIZE[0]))
                        yy = int(math.floor(j/MAP_SIZE[1]))
                        avoidblocks.append((xx, yy))
        
        # Avoid the colliding Bot
        cx, cy = bot.getLocation()
        w, h = bot.getSize()
        xRange = (cx - w/2 - MAP_SIZE[0]/2, cx + w/2 + MAP_SIZE[0]/2)
        yRange = (cy - h/2 - MAP_SIZE[1]/2, cy + h/2 + MAP_SIZE[1]/2)
        for i in range(int(xRange[0]), int(xRange[1])):
            for j in range(int(yRange[0]), int(yRange[1])):
                xx = int(math.floor(i/MAP_SIZE[0]))
                yy = int(math.floor(j/MAP_SIZE[1]))
                avoidblocks.append((xx, yy))

        return list(set(avoidblocks))

    # Get the location of the bot
    def getLocation(self):
        return self.x, self.y
    
    # Get the size of the bot
    def getSize(self):
        return self.size
    
    # Get the number of orders assigned to the bot
    def getNoOfOrders(self):
        return len(self.orders)
    
    # Get the distance from the kitchen to the position of bot after last order
    def lastOrderToKitchen(self):
        if self.getNoOfOrders() > 1:
            return self.distanceTo(self.orders[-1].getTable())
        elif self.currentOrder:
            return self.distanceTo(self.currentOrder.getTable())
        else:
            return 0
        
    # Assign order to the bot
    def assignOrder(self, order):
        self.orders.append(order)

    # Go towards charger
    def setTowardsCharger(self, towardsCharger):
        self.towardsCharger = towardsCharger

    # Get Charger
    def getCharger(self, passiveObjects):
        for obj in passiveObjects:
            if isinstance(obj, Charger):
                return obj
    
    # Get Kitchen
    def getKitchen(self, passiveObjects):
        for obj in passiveObjects:
            if isinstance(obj, Kitchen):
                return obj
            
    # Get the path of the bot
    def getPath(self):
        return self.path
    
    # Remove path
    def endPath(self):
        self.path = None
        
    # Sense distance to the kitchen
    def senseKitchen(self, passiveObjects):
        kitchenL = 0.0
        kitchenR = 0.0
        for pp in passiveObjects:
            if isinstance(pp,Kitchen):
                lx,ly = pp.getLocation()
                distanceL = math.sqrt( (lx-self.sensorPositions[0])*(lx-self.sensorPositions[0]) + \
                                       (ly-self.sensorPositions[1])*(ly-self.sensorPositions[1]) )
                distanceR = math.sqrt( (lx-self.sensorPositions[2])*(lx-self.sensorPositions[2]) + \
                                       (ly-self.sensorPositions[3])*(ly-self.sensorPositions[3]) )
                kitchenL += 200000/(distanceL*distanceL)
                kitchenR += 200000/(distanceR*distanceR)
        return kitchenL, kitchenR

    # Sense the distance to the table
    def senseTable(self, table):
        tableL = 0.0
        tableR = 0.0

        lx,ly = table.getLocation()
        distanceL = math.sqrt( (lx-self.sensorPositions[0])*(lx-self.sensorPositions[0]) + \
                            (ly-self.sensorPositions[1])*(ly-self.sensorPositions[1]) )
        distanceR = math.sqrt( (lx-self.sensorPositions[2])*(lx-self.sensorPositions[2]) + \
                            (ly-self.sensorPositions[3])*(ly-self.sensorPositions[3]) )
        tableL += 200000/(distanceL*distanceL)
        tableR += 200000/(distanceR*distanceR)

        return tableL, tableR

    # Sense the distance to the Charger
    def senseChargers(self, passiveObjects):
        chargerL = 0.0
        chargerR = 0.0
        for pp in passiveObjects:
            if isinstance(pp,Charger):
                lx,ly = pp.getLocation()
                distanceL = math.sqrt( (lx-self.sensorPositions[0])*(lx-self.sensorPositions[0]) + \
                                       (ly-self.sensorPositions[1])*(ly-self.sensorPositions[1]) )
                distanceR = math.sqrt( (lx-self.sensorPositions[2])*(lx-self.sensorPositions[2]) + \
                                       (ly-self.sensorPositions[3])*(ly-self.sensorPositions[3]) )
                chargerL += 200000/(distanceL*distanceL)
                chargerR += 200000/(distanceR*distanceR)
        return chargerL, chargerR

    # Get distance to a specific object
    def distanceTo(self,obj):
        xx,yy = obj.getLocation()
        return math.sqrt( math.pow(self.x-xx,2) + math.pow(self.y-yy,2) )
    
    # Get distance to a specific point
    def distanceToPoint(self,xx, yy):
        return math.sqrt( math.pow(self.x-xx,2) + math.pow(self.y-yy,2) )

    # Get distance from right sensor
    def distanceToRightSensor(self,lx,ly):
        return math.sqrt( (lx-self.sensorPositions[0])*(lx-self.sensorPositions[0]) + \
                          (ly-self.sensorPositions[1])*(ly-self.sensorPositions[1]) )

    # Get distance from left sensor
    def distanceToLeftSensor(self,lx,ly):
        return math.sqrt( (lx-self.sensorPositions[2])*(lx-self.sensorPositions[2]) + \
                            (ly-self.sensorPositions[3])*(ly-self.sensorPositions[3]) )

    # Update bot state at each timestep
    def update(self,canvas,passiveObjects,dt):

        if self.battery == BATTERY_CAPACITY and len(self.orders) == 0 and not self.currentOrder:
            return
        self.battery -= 1 # Reduce battery after every timestep
        
        # Increase battery if the bot is close to the charger
        for rr in passiveObjects:
            if isinstance(rr,Charger) and self.distanceTo(rr)<80:
                if self.battery <  BATTERY_CAPACITY - 10:
                    self.battery += 10
                else:
                    self.battery = BATTERY_CAPACITY
        
        # Battery can never be negative
        if self.battery<=0:
            self.battery = 0
        self.move(canvas,dt)

    # Draw the robot at its current position
    def draw(self,canvas):
        
        points = [ (self.x + (self.size[0]/2)*math.sin(self.theta)) - (self.size[0]/2)*math.sin((math.pi/2.0)-self.theta), \
                   (self.y - (self.size[1]/2)*math.cos(self.theta)) - (self.size[1]/2)*math.cos((math.pi/2.0)-self.theta), \
                   (self.x - (self.size[0]/2)*math.sin(self.theta)) - (self.size[0]/2)*math.sin((math.pi/2.0)-self.theta), \
                   (self.y + (self.size[1]/2)*math.cos(self.theta)) - (self.size[1]/2)*math.cos((math.pi/2.0)-self.theta), \
                   (self.x - (self.size[0]/2)*math.sin(self.theta)) + (self.size[0]/2)*math.sin((math.pi/2.0)-self.theta), \
                   (self.y + (self.size[1]/2)*math.cos(self.theta)) + (self.size[1]/2)*math.cos((math.pi/2.0)-self.theta), \
                   (self.x + (self.size[0]/2)*math.sin(self.theta)) + (self.size[0]/2)*math.sin((math.pi/2.0)-self.theta), \
                   (self.y - (self.size[1]/2)*math.cos(self.theta)) + (self.size[1]/2)*math.cos((math.pi/2.0)-self.theta)  \
                ]
        canvas.create_polygon(points, fill="turquoise4", tags=self.name)

        self.sensorPositions = [ (self.x + (self.size[0]/3)*math.sin(self.theta)) + (self.size[0]/2)*math.sin((math.pi/2.0)-self.theta), \
                                 (self.y - (self.size[1]/3)*math.cos(self.theta)) + (self.size[1]/2)*math.cos((math.pi/2.0)-self.theta), \
                                 (self.x - (self.size[0]/3)*math.sin(self.theta)) + (self.size[0]/2)*math.sin((math.pi/2.0)-self.theta), \
                                 (self.y + (self.size[1]/3)*math.cos(self.theta)) + (self.size[1]/2)*math.cos((math.pi/2.0)-self.theta)  \
                            ]
    
        centre1PosX = self.x 
        centre1PosY = self.y
        
        # Decide the top look of the bot
        if self.battery < MIN_BATTERY:
            plateColor = "red"
        else:
            plateColor = "PaleTurquoise1"

        if self.towardsTable:
            canvas.create_image(centre1PosX, centre1PosY, image = self.foodImage, tags = self.name)
        else:
            canvas.create_oval(centre1PosX-(self.size[0]/4),centre1PosY-(self.size[1]/4),\
                            centre1PosX+(self.size[0]/4),centre1PosY+(self.size[1]/4),\
                            fill=plateColor,tags=self.name)
        
        # Update the battery level display
        canvas.create_text(40,10,text='Battery: ', font=("Arial", 12), tags=self.name)
        canvas.create_text((self.id + 1) * 120, 10, text = self.name + ": " + str(self.battery), font=("Arial", 12), tags = self.name)

        wheel1PosX = self.x - (self.size[0]/2)*math.sin(self.theta)
        wheel1PosY = self.y + (self.size[1]/2)*math.cos(self.theta)
        canvas.create_oval(wheel1PosX-(self.size[0]/20),wheel1PosY-(self.size[1]/20),\
                                         wheel1PosX+(self.size[0]/20),wheel1PosY+(self.size[1]/20),\
                                         fill="red",tags=self.name)

        wheel2PosX = self.x + (self.size[0]/2)*math.sin(self.theta)
        wheel2PosY = self.y - (self.size[1]/2)*math.cos(self.theta)
        canvas.create_oval(wheel2PosX-(self.size[0]/20),wheel2PosY-(self.size[1]/20),\
                                         wheel2PosX+(self.size[0]/20),wheel2PosY+(self.size[1]/20),\
                                         fill="green",tags=self.name)

        sensor1PosX = self.sensorPositions[0]
        sensor1PosY = self.sensorPositions[1]
        sensor2PosX = self.sensorPositions[2]
        sensor2PosY = self.sensorPositions[3]
        canvas.create_oval(sensor1PosX-(self.size[0]/20),sensor1PosY-(self.size[1]/20), \
                           sensor1PosX+(self.size[0]/20),sensor1PosY+(self.size[1]/20), \
                           fill="yellow",tags=self.name)
        canvas.create_oval(sensor2PosX-(self.size[0]/20),sensor2PosY-(self.size[1]/20), \
                           sensor2PosX+(self.size[0]/20),sensor2PosY+(self.size[1]/20), \
                           fill="yellow",tags=self.name)

    # Handle the movement
    def move(self,canvas,dt):
        if self.battery==0:
            self.sl = 0
            self.sl = 0
        if self.sl==self.sr:
            R = 0
        else:
            R = (self.ll/2.0)*((self.sr+self.sl)/(self.sl-self.sr))
        omega = (self.sl-self.sr)/self.ll
        ICCx = self.x-R*math.sin(self.theta) #instantaneous centre of curvature
        ICCy = self.y+R*math.cos(self.theta)
        m = np.matrix( [ [math.cos(omega*dt), -math.sin(omega*dt), 0], \
                        [math.sin(omega*dt), math.cos(omega*dt), 0],  \
                        [0,0,1] ] )
        v1 = np.matrix([[self.x-ICCx],[self.y-ICCy],[self.theta]])
        v2 = np.matrix([[ICCx],[ICCy],[omega*dt]])
        newv = np.add(np.dot(m,v1),v2)
        newX = newv.item(0)
        newY = newv.item(1)
        newTheta = newv.item(2)
        newTheta = newTheta%(2.0*math.pi) #make sure angle doesn't go outside [0.0,2*pi)
        self.x = newX
        self.y = newY
        self.theta = newTheta        
        if self.sl==self.sr: # straight line movement
            self.x += self.sr*math.cos(self.theta) #sr wlog
            self.y += self.sr*math.sin(self.theta)
        canvas.delete(self.name)
        self.draw(canvas)

    # Get the minimum distance allowed from an object
    def minDistanceFrom(self, obj):
        diagonalSize = math.sqrt( math.pow(self.size[0],2) + math.pow(self.size[1],2) )
        objSize = obj.getSize()
        objDiagonalSize = math.sqrt( math.pow(objSize[0],2) + math.pow(objSize[1],2) )

        return (diagonalSize/2 + objDiagonalSize/2)
    
    # Get the distance of a specific object from an expected position
    def expectedDistance(self, sl, sr, obj):
        expectedx, expectedy = self.expectedPosition(sl, sr)
        xx, yy = obj.getLocation()
        return math.sqrt( math.pow(expectedx-xx,2) + math.pow(expectedy-yy,2) )
    
    # Get the expected position after moving
    def expectedPosition(self, sl, sr, dt = 1):
        if sl==sr:
            R = 0
        else:
            R = (self.ll/2.0)*((sr+sl)/(sl-sr))
        omega = (sl-sr)/self.ll
        ICCx = self.x-R*math.sin(self.theta) #instantaneous centre of curvature
        ICCy = self.y+R*math.cos(self.theta)
        m = np.matrix( [ [math.cos(omega*dt), -math.sin(omega*dt), 0], \
                        [math.sin(omega*dt), math.cos(omega*dt), 0],  \
                        [0,0,1] ] )
        v1 = np.matrix([[self.x-ICCx],[self.y-ICCy],[self.theta]])
        v2 = np.matrix([[ICCx],[ICCy],[omega*dt]])
        newv = np.add(np.dot(m,v1),v2)
        newX = newv.item(0)
        newY = newv.item(1)
        newTheta = newv.item(2)
        newTheta = newTheta%(2.0*math.pi) #make sure angle doesn't go outside [0.0,2*pi)
        updatedx = newX
        updatedy = newY
        updatedtheta = newTheta        
        if sl==sr: # straight line movement
            updatedx += sr*math.cos(updatedtheta) #sr wlog
            updatedy += sr*math.sin(updatedtheta)
        return (updatedx, updatedy)

    # Detect collision with a customer
    def collisionWithCustomer(self,agents):
        for rr in agents:
            if isinstance(rr,CustomerGroup):
                if not rr.isSitting():
                    for customer in rr.getCustomers():              
                        if self.distanceTo(customer)<self.minDistanceFrom(customer):
                            return (True, customer)
        return (False, None)
    
    # Detect collision with another bot
    def collisionWithBot(self, agents):
        for rr in agents:
            if isinstance(rr, Bot) and not rr == self:
                if self.distanceTo(rr) < self.minDistanceFrom(rr):
                    return (True, rr)
        return (False, None)
    
    # Detect collision with a passive object
    def collisionWithPassiveObject(self, passiveObjects):
        for obj in passiveObjects:
            # Collision with a table
            if isinstance(obj, Table):
                if self.currentOrder and not obj == self.currentOrder.getTable():
                    if self.distanceTo(obj) < self.minDistanceFrom(obj):
                        return (True, obj)
            # Collision with a charger
            elif isinstance(obj, Charger):
                if not self.towardsCharger:
                    if self.distanceTo(obj) < self.minDistanceFrom(obj):
                        return (True, obj)
            # collision with Kitchen
            elif isinstance(obj, Kitchen):
                if not self.towardsKitchen:
                    if self.distanceTo(obj) < self.minDistanceFrom(obj):
                        return (True, obj)
            else:
                if self.distanceTo(obj) < self.minDistanceFrom(obj):
                    return (True, obj)

        return (False, None)
    
# Manager to assign duties to manage orders and assign orders to bots
class Manager:
    orders = []
    assignedOrders = []

    # Get customer order
    @classmethod
    def noteOrder(self, order):
        Manager.orders.append(order)

    @classmethod
    def thinkAndAct(self, agents):
        # assign orders and remove already completed orders
        for order in Manager.orders:
            def bot_comparison(bot):
                return (bot.getNoOfOrders(), bot.lastOrderToKitchen())

            availableBots = [agent for agent in agents[::-1] if (isinstance(agent, Bot))]
            bot = min(availableBots, key=bot_comparison)
            bot.assignOrder(order)
            Manager.orders.remove(order)
            Manager.assignedOrders.append(order)

        for order in Manager.assignedOrders:
            if order.isDelivered():
                Manager.assignedOrders.remove(order)

    def update():
        pass

# Group of Customers
class CustomerGroup:
    lastgroupNo = 0
    waitingTimes = []

    # Attributes of a group of customers
    def __init__(self, canvas, noOfCustomers, door, tables):
        CustomerGroup.lastgroupNo += 1
        self.groupNo = CustomerGroup.lastgroupNo
        self.number = noOfCustomers
        self.door = door
        self.x, self.y = door.getLocation()
        self.wait = True
        self.tables = tables
        self.waitingTime = -1
        self.orderPlaced = False
        self.orderReceived = False
        self.order = None

        self.customers = [Customer("Customer" + str(self.groupNo) + '_' + str(i), canvas, self.x, self.y) for i in range(noOfCustomers)]

    # Note the waiting time of the customer group once it leaves the restaurant
    def __del__(self):
        if self.orderReceived:
            CustomerGroup.waitingTimes.append(self.waitingTime)
            print(CustomerGroup.getWaitingTimes())
        
    # Get the waiting times of the customer groups
    @classmethod
    def getWaitingTimes(self):
        return CustomerGroup.waitingTimes

    # Get all the customers in the customer group
    def getCustomers(self):
        return self.customers
    
    # Is the customer group sitting at a table?
    def isSitting(self):
        for customer in self.customers:
            if not customer.isSitting():
                return False
        return True
    
    # Place an order
    def giveOrder(self):
        self.order =  Order('Order ' + str(self.groupNo), self.table)
        Manager.noteOrder(self.order)

    # Receive order from the bot
    def receiveOrder(self):
        self.orderReceived = True

    # Leave the restaurant through the door
    def leaveRestaurant(self):
        for customer in self.customers:
            customer.leave(self.door)

    # Check if all the customers in the customer group have left the restaurant
    def hasLeft(self):
        for customer in self.customers:
            if not customer.leftRestaurant():
                return False
        return True

    # Draw the customers in the customer group
    def draw(self, canvas):
        for customer in self.customers:
            customer.draw(canvas)

    # Manage the behavior 
    def thinkAndAct(self, agents, passiveObjects, canvas):
        # Delete the customer group agent once it has left the restaurant
        if self.hasLeft():
            agents.remove(self)
            for customer in self.customers:
                del customer
            del self
            return

        # Wait for a table to be assigned
        if self.wait:
            self.waitingTime += 1          
            # Get all the available tables with the required capacity  
            if self.number <= 4:
                availableTables = [table for table in self.tables if (table.capacity == 4 and not table.isOccupied())]
                if not availableTables:
                    availableTables = [table for table in self.tables if (table.capacity == 6 and not table.isOccupied())]
            else:
                availableTables = [table for table in self.tables if (table.capacity == 6 and not table.isOccupied())]
            
            # Choose one of the available tables at random
            if availableTables:
                self.wait = False
                self.table = random.choice(availableTables)
                print("no of customers: ", str(self.number))
                print(self.table.getLocation())

                # Assign the table to each customer
                for customer in self.customers:
                    if not customer.tableAssigned():
                        customer.assignTable(self.table)
        
        # Handle the different states of the custoemr group

        elif self.isSitting() and not self.orderPlaced:
            self.giveOrder()
            self.orderPlaced = True

        elif self.orderPlaced and not self.orderReceived:
            if self.order.isDelivered():
                print('Deliveredddddddd', self.groupNo)
                print(self.waitingTime)
                self.orderReceived = True
            else:
                self.waitingTime += 1

        elif self.orderReceived:
            self.leaveRestaurant()

        for customer in self.customers:
            customer.thinkAndAct(agents,passiveObjects,canvas)

    # Update the state of the customer group
    def update(self,canvas,passiveObjects,dt):
        for customer in self.customers:
            customer.update(canvas, passiveObjects, dt)

# Customer class
class Customer:
    # Customer attributes
    def __init__(self,namep,canvasp,x,y):
        self.x = x
        self.y = y
        self.theta = random.uniform(0.0,2.0*math.pi)
        self.name = namep
        self.canvas = canvasp
        self.table = None
        self.path = None
        self.sitting = False
        self.leaving = False
        self.updateAngle = 0
        self.size = (20, 20)
        self.step = 2
        self.door = None
        self.hasLeft = False
        imgFile = Image.open("images/customer.png")
        imgFile = imgFile.resize(self.size, Image.LANCZOS)
        self.image = ImageTk.PhotoImage(imgFile)

    # Draw the customer
    def draw(self,canvas):
        canvas.create_image(self.x,self.y,image=self.image,tags=self.name)

    # Get location of the customer
    def getLocation(self):
        return self.x, self.y
    
    # Get size of the customer
    def getSize(self):
        return self.size

    # Get the table assigned to the customer
    def tableAssigned(self):
        return self.table
    
    # Check if the customer is sitting
    def isSitting(self):
        return self.sitting
    
    # Check if the customer has left the restaurant
    def leftRestaurant(self):
        return self.hasLeft
    
    # Assign table
    def assignTable(self, table):
        self.table = table
        self.table.occupy()

    # Sit at the assigned table
    def sit(self):
        self.x = self.table.getLocation()[0]
        self.y = self.table.getLocation()[1]
        self.table.fillASeat()
        self.table.updateOccupiedQuantity()
        self.canvas.delete(self.name)
        self.draw(self.canvas)

    # Leave the restaurant
    def leave(self, door):
        self.sitting = False
        self.leaving = True
        self.door = door
        self.table.emptyTable()
        self.table.updateOccupiedQuantity()

    # Handle behavior of the customer
    def thinkAndAct(self, agents, passiveObjects, canvas):
        if self.sitting:
            return
        
        if not self.table:  # If table is not assigned, do nothing
            return
        
        # Decide the angle
        if self.leaving:
            self.desiredAngle = self.angleTo(self.door)
        else:
            self.desiredAngle = self.angleTo(self.table)

        tableCollision = False
        
        # Handle collision with passive objects
        while self.collisionWithPassiveObject(agents, passiveObjects):
            self.desiredAngle += math.radians(self.updateAngle)
            tableCollision = True
        
        # If not colliding with a table
        if not tableCollision:
            botcollision = False
            initialDesire = self.desiredAngle
            # Handle collision with a bot
            while self.collisionWithBot(agents, passiveObjects):
                self.desiredAngle += math.radians(self.updateAngle)
                if math.degrees(self.desiredAngle - initialDesire) > 270:
                    print('more than 270 rotation')
                    break
                botcollision = True
            if not botcollision:
                # Handle collision with another customer
                if self.collisionWithCustomer(agents, passiveObjects):
                    self.desiredAngle += math.radians(self.updateAngle)
        
        self.theta = self.desiredAngle

    # Update customer state after every timestep
    def update(self,canvas,passiveObjects,dt):
        if self.sitting:
            return
        if not self.table:
            return
        if self.hasLeft:
            canvas.delete(self.name)
        else:
            self.move(canvas,dt)

    # Get distance to an object
    def distanceTo(self,obj):
        xx,yy = obj.getLocation()
        return math.sqrt( math.pow(self.x-xx,2) + math.pow(self.y-yy,2) )
    
    # Get angle to an object
    def angleTo(self, obj):
        objLoc = obj.getLocation()
        dx = objLoc[0] - self.x
        dy = self.y - objLoc[1]

        return math.atan2(dy, dx)
    
    # Get minimum distance allowed from an object
    def minDistanceFrom(self, obj):
        objSize = obj.getSize()
        objDiagonalSize = math.sqrt( math.pow(objSize[0],2) + math.pow(objSize[1],2) )

        return (self.size[0] + objDiagonalSize/2)
    
    # Check collision with a passive object
    def collisionWithPassiveObject(self, agents, passiveObjects):
        self.updateAngle = random.randint(25, 45)

        for obj in passiveObjects:
            # Collision with a table
            if isinstance(obj, Table):
                if self.distanceTo(obj) < self.minDistanceFrom(obj):
                    angleToObj = self.angleTo(obj)
                    if abs(self.desiredAngle - angleToObj) < math.radians(90):
                        if not self.leaving and obj == self.table:
                            self.sit()
                            self.sitting = True
                            
                            return False
                        if self.desiredAngle < angleToObj:
                            self.updateAngle = 0 - self.updateAngle
                            
                        return True
            # Collision with Kitchen
            elif isinstance(obj, Kitchen):
                if self.distanceTo(obj) < KITCHEN_SIZE[0]/2:
                    if abs(self.desiredAngle - self.angleTo(obj)) < math.radians(90):
                        return True
            # Collision with the door
            elif isinstance(obj, Door):
                if obj == self.door:
                    if self.distanceTo(obj) < DOOR_SIZE[1]/2:
                        self.hasLeft = True
                      
        return False

    # Check collision with another customer
    def collisionWithCustomer(self, agents, passiveObjects):
        self.updateAngle = 90

        for agent in agents:
            if isinstance(agent, CustomerGroup):
                if agent.isSitting():
                    continue
                for customer in agent.getCustomers():
                    if self == customer or customer.isSitting():
                        continue
                    if self.distanceTo(customer) < self.minDistanceFrom(customer):
                        if abs(self.desiredAngle - self.angleTo(customer)) < math.radians(90):
                            
                            self.updateAngle = 0 - random.randint(0, self.updateAngle)
                            return True
        return False
    
    # Check collision with a bot
    def collisionWithBot(self, agents, passiveObjects):
        self.updateAngle = 90
        for agent in agents:
            if isinstance(agent, Bot):
                if self.distanceTo(agent) < self.minDistanceFrom(agent):
                    if abs(self.desiredAngle - self.angleTo(agent)) < math.radians(90):
                        self.updateAngle = random.randint(0, self.updateAngle)
                        return True  
        return False              

    # Handle movement
    def move(self,canvas,dt):
                
        if self.theta < 0:
            self.theta += (2*math.pi)
        

        if START_ANGLE < self.theta <= START_ANGLE + ANGLE_RANGE:
            self.x += self.step
            self.y -= self.step/2
        elif START_ANGLE + ANGLE_RANGE < self.theta <= START_ANGLE + 2*ANGLE_RANGE:
            self.x += self.step
            self.y -= self.step
        elif START_ANGLE + 2*ANGLE_RANGE < self.theta <= START_ANGLE + 3*ANGLE_RANGE:
            self.x += self.step/2
            self.y -= self.step
        elif START_ANGLE + 3*ANGLE_RANGE < self.theta <= START_ANGLE + 4*ANGLE_RANGE:
            self.y -= self.step
        elif START_ANGLE + 4*ANGLE_RANGE < self.theta <= START_ANGLE + 5*ANGLE_RANGE:
            self.x -= self.step/2
            self.y -= self.step
        elif START_ANGLE + 5*ANGLE_RANGE < self.theta <= START_ANGLE + 6*ANGLE_RANGE:
            self.x -= self.step
            self.y -= self.step
        elif START_ANGLE + 6*ANGLE_RANGE < self.theta <= START_ANGLE + 7*ANGLE_RANGE:
            self.x -= self.step
            self.y -= self.step/2
        elif START_ANGLE + 7*ANGLE_RANGE < self.theta <= START_ANGLE + 8*ANGLE_RANGE:
            self.x -= self.step
        elif START_ANGLE + 8*ANGLE_RANGE < self.theta <= START_ANGLE + 9*ANGLE_RANGE:
            self.x -= self.step
            self.y += self.step/2
        elif START_ANGLE + 9*ANGLE_RANGE < self.theta <= START_ANGLE + 10*ANGLE_RANGE:
            self.x -= self.step
            self.y += self.step
        elif START_ANGLE + 10*ANGLE_RANGE < self.theta <= START_ANGLE + 11*ANGLE_RANGE:
            self.x -= self.step/2
            self.y += self.step
        elif START_ANGLE + 11*ANGLE_RANGE < self.theta <= START_ANGLE + 12*ANGLE_RANGE:
            self.y += self.step
        elif START_ANGLE + 12*ANGLE_RANGE < self.theta <= START_ANGLE + 13*ANGLE_RANGE:
            self.x += self.step/2
            self.y += self.step
        elif START_ANGLE + 13*ANGLE_RANGE < self.theta <= START_ANGLE + 14*ANGLE_RANGE:
            self.x += self.step
            self.y += self.step
        elif START_ANGLE + 14*ANGLE_RANGE < self.theta <= START_ANGLE + 15*ANGLE_RANGE:
            self.x += self.step
            self.y += self.step/2
        elif (START_ANGLE + 15*ANGLE_RANGE < self.theta) or self.theta <= START_ANGLE:
            self.x += self.step

        # Avoiding going out of the screen
        if self.x<20:
            self.x = 20
        if self.x>WINDOW_SIZE[0] - 20:
            self.x = WINDOW_SIZE[0] - 20
        if self.y<20:
            self.y=20
        if self.y>WINDOW_SIZE[1] - 20:
            self.y = WINDOW_SIZE[1] - 20
        
        canvas.delete(self.name)
        self.draw(canvas)
        
# Restaurant table
class Table:
    # Attributes
    def __init__(self, namep, width, height, xp, yp, capacity):
        self.width = width
        self.height = height
        self.centreX = xp
        self.centreY = yp
        self.name = namep
        self.capacity = capacity
        self.occupied = False
        self.seatsFilled = 0

    # Draw table
    def draw(self, canvas):
        self.canvas = canvas
        x = self.width / 2
        y = self.height / 2

        # Draw table body
        body = canvas.create_rectangle(
            self.centreX - x, self.centreY - y,
            self.centreX + x, self.centreY + y,
            fill="brown", tags=self.name
        )

        # Draw table mats
        mat_width = min(self.width, self.height) / 3
        mat_height = min(self.width, self.height) / 6

        # Calculate the number of mats to draw based on capacity
        num_mats = min(self.capacity, 6)  

        # Draw mats in front of where seats would be
        angle_step = 360 / self.capacity
        for i in range(num_mats):
            angle_rad = (90 + i * angle_step) * (3.14159 / 180)  # Convert degrees to radians

            if (i == 0 or i == self.capacity/2):
                mat_x1 = self.centreX + x * 0.8 * math.cos(angle_rad) - mat_width / 2
                mat_x2 = mat_x1 + mat_width
                mat_y1 = self.centreY + y * 0.8 * math.sin(angle_rad) - mat_height / 2
                mat_y2 = mat_y1 + mat_height
            else:
                mat_x1 = self.centreX + x * 0.8 * math.cos(angle_rad) - mat_height / 2
                mat_x2 = mat_x1 + mat_height
                mat_y1 = self.centreY + y * 0.8 * math.sin(angle_rad) - mat_width / 2
                mat_y2 = mat_y1 + mat_width

            mat = canvas.create_rectangle(
                mat_x1, mat_y1, mat_x2, mat_y2,
                fill="gray", outline=""
            )
        # Number of customers sitting at the table
        self.quantityTag = canvas.create_text(self.centreX + 20, self.centreY, text="", font=("Arial", 12), fill="black")
    
    # Get location of table
    def getLocation(self):
        return self.centreX, self.centreY
    
    # Get size of the table
    def getSize(self):
        return (self.width, self.height)
    
    # Table occupied
    def occupy(self):
        self.occupied = True

    # Empty the table
    def emptyTable(self):
        self.occupied = False
        self.seatsFilled = 0

    # Return occupied state
    def isOccupied(self):
        return self.occupied
    
    # Fill a seat if a customer arrives at the table
    def fillASeat(self):
        self.seatsFilled += 1

    # Return number of seats filled
    def getSeatsFilled(self):
        return self.seatsFilled
    
    # Update the number of customers tag on the table
    def updateOccupiedQuantity(self):
        if self.seatsFilled > 0:
            self.canvas.itemconfigure(self.quantityTag, text=str(self.seatsFilled))
        else:
            self.canvas.itemconfigure(self.quantityTag, text="")
    
# Kitchen for the bot to collect food from
class Kitchen:
    # Kitchen attributes
    def __init__(self,namep,width,height,xp,yp):
        self.width = width
        self.height = height
        self.centreX = xp
        self.centreY = yp
        self.name = namep
        
    # Draw the kitchen
    def draw(self,canvas):
        x = self.width/2
        y = self.height/2
        body = canvas.create_rectangle(self.centreX-x,self.centreY-y, \
                                  self.centreX+x,self.centreY+y, \
                                  fill="green",tags=self.name)
        
        canvas.create_text(self.centreX, self.centreY, text="Kitchen", font=("Arial", 12), fill="black")

    # Get location of the kitchen
    def getLocation(self):
        return self.centreX, self.centreY
    
    # Get size of the kitchen
    def getSize(self):
        return (self.width, self.height)
    
# Charger for the bots
class Charger:
    # Charger attributes
    def __init__(self,namep,width,height,xp,yp):
        self.width = width
        self.height = height
        self.centreX = xp
        self.centreY = yp
        self.name = namep
        
    # Draw the charger
    def draw(self,canvas):
        x = self.width/2
        y = self.height/2
        body = canvas.create_rectangle(self.centreX-x,self.centreY-y, \
                                  self.centreX+x,self.centreY+y, \
                                  fill="yellow",tags=self.name)
        
        canvas.create_text(self.centreX, self.centreY, text="Charger", font=("Arial", 10), fill="black")

    # Get location of the charger
    def getLocation(self):
        return self.centreX, self.centreY
    
    # Get size of the charger
    def getSize(self):
        return (self.width, self.height)
    
# Door to enter and leave the restaurant
class Door:
    # Door attributes
    def __init__(self,namep,width,height,xp,yp):
        self.width = width
        self.height = height
        self.centreX = xp
        self.centreY = yp
        self.name = namep
        
    # Draw the door
    def draw(self,canvas):
        x = self.width/2
        y = self.height/2
        body = canvas.create_rectangle(self.centreX-x,self.centreY-y, \
                                  self.centreX+x,self.centreY+y, \
                                  fill="grey",tags=self.name)
    
    # Get door location
    def getLocation(self):
        return self.centreX, self.centreY
    
    # Get size of the door
    def getSize(self):
        return (self.width, self.height)

# Initialize the tkinter window
def initialise(window):
    window.resizable(False,False)
    canvas = tk.Canvas(window,width=WINDOW_SIZE[0],height=WINDOW_SIZE[1])
    canvas.pack()
    return canvas

# Create all the passive and active objects
def createObjects(canvas,noOfBots, method):
    agents = []
    passiveObjects = []
    
    # Create the bots
    for i in range(noOfBots):
        x = WINDOW_SIZE[0] - (i + 1) * CHARGER_SIZE[0] - BOT_SIZE[0]
        y = WINDOW_SIZE[1] - BOT_SIZE[1]/2
        bot = Bot("Bot"+str(i), x, y, canvas, method, i)
        brain = Brain(bot)
        bot.setBrain(brain)
        agents.append(bot)
        bot.draw(canvas)

    # Create the manager
    manager = Manager()

    # Create the tables
    cellsize = (int(TABLE_SIZE[0]*2.8), int(TABLE_SIZE[1]*2.3))
    for i in range(WINDOW_SIZE[0]//cellsize[0]):
        for j in range(WINDOW_SIZE[1]//cellsize[1] - 1):
            capacity = 4
            if (0 < i < 4 and j == 0) or (i == 2 and j == 1):
                capacity = 6
            table = Table("Table" + str(i) + '_' + str(j), TABLE_SIZE[0], TABLE_SIZE[1], i * cellsize[0] + cellsize[0]/2, j * cellsize[1] + cellsize[1]/2, capacity)
            passiveObjects.append(table)
            table.draw(canvas)

    # Create the Kitchen
    kitchen = Kitchen("Kitchen", KITCHEN_SIZE[0], KITCHEN_SIZE[1], WINDOW_SIZE[0]/2, WINDOW_SIZE[1] - KITCHEN_SIZE[1]/2)
    passiveObjects.append(kitchen)
    kitchen.draw(canvas)

    # Create the charger
    charger = Charger("Charger", CHARGER_SIZE[0], CHARGER_SIZE[1], WINDOW_SIZE[0] - CHARGER_SIZE[0]/2, WINDOW_SIZE[1] - CHARGER_SIZE[1]/2)
    passiveObjects.append(charger)
    charger.draw(canvas)

    # Create the doors
    door1 = Door("Door1", DOOR_SIZE[0], DOOR_SIZE[1], DOOR_SIZE[0]/2, WINDOW_SIZE[1] - KITCHEN_SIZE[1] - DOOR_SIZE[1])
    passiveObjects.append(door1)
    door1.draw(canvas)
    door2 = Door("Door2", DOOR_SIZE[0], DOOR_SIZE[1], WINDOW_SIZE[0] - DOOR_SIZE[0]/2, WINDOW_SIZE[1] - KITCHEN_SIZE[1] - DOOR_SIZE[1])
    passiveObjects.append(door2)
    door2.draw(canvas)
    
    return agents, passiveObjects, manager

# Move the created objects after every timestep
def moveIt(canvas,agents,passiveObjects, manager):
    for rr in agents:
        rr.thinkAndAct(agents,passiveObjects,canvas)
        rr.update(canvas,passiveObjects,1.0)

    manager.thinkAndAct(agents)
    
    # Customers enter the restaurant with a probability of 0.005 at each timestep
    if random.random() < 0.005:
        doors = [obj for obj in passiveObjects if isinstance(obj, Door)]
        tables = [obj for obj in passiveObjects if isinstance(obj, Table)]
        customerGroup = CustomerGroup(canvas, random.randint(1, 6), random.choice(doors), tables)
        agents.append(customerGroup)
        customerGroup.draw(canvas)
    # Update the agents and objects after every 25ms
    canvas.after(30,moveIt,canvas,agents,passiveObjects,manager)

# Simulate the function to move all the created objects
def simulateMoveIt(canvas, agents, passiveObjects, manager):
    for rr in agents:
        rr.thinkAndAct(agents, passiveObjects, canvas)
        rr.update(canvas, passiveObjects, 1.0)

    manager.thinkAndAct(agents)

    # Customers enter the restaurant with a probability of 0.005 at each timestep
    if random.random() < 0.005:
        doors = [obj for obj in passiveObjects if isinstance(obj, Door)]
        tables = [obj for obj in passiveObjects if isinstance(obj, Table)]
        customerGroup = CustomerGroup(canvas, random.randint(1, 6), random.choice(doors), tables)
        agents.append(customerGroup)
    time.sleep(0.01)

# Simulate the restaurant environment without GUI
def simulate(duration = 200, noOfBots = 1, method = "astar"):
    window = tk.Tk()
    canvas = initialise(window)
    agents, passiveObjects, manager = createObjects(canvas, noOfBots, method)
    moves = 0
    
    # Run the simulation for the specified time duration
    start_time = time.time()
    while time.time() - start_time < duration:
        simulateMoveIt(canvas, agents, passiveObjects, manager)
        moves += 1

    # Get the waiting times of each customer group
    waitingTimes = CustomerGroup.getWaitingTimes()
    # Destroy the window after simulation is completed
    window.destroy()
    # Reinitialize the customer waiting times for the next simulation
    CustomerGroup.waitingTimes = []
    # Return the waiting times
    return waitingTimes

# Main function to run the restaurant simulation
def main():
    window = tk.Tk()
    canvas = initialise(window)
    agents, passiveObjects, manager = createObjects(canvas,noOfBots=2, method = 'astar')
    moveIt(canvas,agents,passiveObjects, manager)
    window.mainloop()
    
if __name__ == "__main__":
    main()