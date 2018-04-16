
# coding: utf-8

# In[14]:


from __future__ import division
get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import misc


# In[15]:


def f(x):
    return np.exp(-x/2) + 10*np.power(x, 2)


# In[16]:


misc.derivative(f, 1.0)


# In[17]:


#Gradient descent
#plot the function
def get_function(x):
    #for point in x:
    x = np.array(x)
    y = np.exp(-x/2) + 10*np.power(x, 2)
    return y


# In[18]:


def get_derivative(x):
    x = np.array(x)
    f_prime = (-1/2)*(np.exp(-x/2)) + 20*x
    return f_prime


# In[19]:


get_derivative(1)


# In[20]:


def draw_plot(points, learning_rate):
    fig1,ax1 = plt.subplots(1)
    fig2,ax2 = plt.subplots(1)
    fig1.set_size_inches(10, 8)
    fig2.set_size_inches(10, 8)
    x1 = np.arange(-1.5, 1.5, 0.001)
    ax1.plot(x1, get_function(x1));
    x2 = np.arange(-1.5, 1.5, 0.001)
    ax2.plot(x2, get_function(x2));
    colors=['green','orange','red']
    #number = str(iteration_time)
    for i in range(3):
        ax1.scatter(points[i], get_function(points[i]),c = colors[i], label = str(i+1));
        ax1.set_ylim((-2, 25))
        m = np.arange(-1.5, 2)
        tangent_line = get_derivative(points[i])*m + (get_function(points[i]) - get_derivative(points[i])*points[i])
            #axes[iteration_time-1].plot(m, tangent_line);
        ax1.plot(m, tangent_line, c = colors[i],label = str(i+1));
        ax1.set_title("Tangent lines for learning rate %s" % (learning_rate))
        ax1.legend()
    ax2.plot(points[:10], get_function(points[:10]), c = "orange")
    ax2.set_title("Gradient descent steps for learning rate %s" % (learning_rate))
    
    
    for j in range(10):
        time = str(j+1)
        ax2.scatter(points[j], get_function(points[j]), label = time);
        ax2.legend()
    fig1.savefig("Tangent lines for learning rate %s." % (learning_rate))
    fig2.savefig("Gradient descent steps for learning rate %s." % (learning_rate))


# In[29]:


def gradient_descent(starting_point, learning_rate):
    magnitude_point = abs(starting_point)
    point = starting_point
    iteration_time = 0
    points = []
    while True:
        points.append(point)
        direction = get_derivative(point)
        #print new_point, get_function(new_point)
        #magnitude_point = abs(direction)
        magnitude_point = np.linalg.norm(direction)
        iteration_time += 1
        if np.isinf(magnitude_point) == True:
            print "The gradient descent does not converge. Change the learning rate."
            draw_plot(points, learning_rate)
            return iteration_time,  point, get_function(point), magnitude_point
            break
        else:
            if magnitude_point < 10**(-10) or iteration_time > 10000:
                draw_plot(points, learning_rate)
                return iteration_time,  point, magnitude_point, get_function(point)
                break
            else:
                new_point = point + learning_rate*(-direction)
                point = new_point


# In[30]:


gradient_descent(1, 0.1)


# In[31]:


gradient_descent(1, 0.01)


# In[24]:


gradient_descent(1, 0.001)


# In[25]:


gradient_descent(1, 0.0001)


# In[28]:


p = 0.024693232270265097
x = np.arange(-10, 10, 0.2)
fig = plt.figure()
fig.set_size_inches(16, 10)
plt.plot(x, get_function(x));
plt.scatter(p, get_function(p));
#ax_arrow = plt.axes()
#plt.arrow(1, get_function(1), 1, get_function(1), \
               #head_width=0.3, head_length=1, fc='k', ec='k');
m = np.arange(-10, 10)
tangent_line = get_derivative(p)*m + (get_function(p) - get_derivative(p)*p)
plt.plot(m, tangent_line);

