import numpy as np
import scipy as scp
from scipy.spatial import distance_matrix

class surface():
	
	def __init__(self,variables):
		
		self.h = np.asarray(variables['h'])
		self.r = np.asarray(variables['r'])
		self.x = np.asarray(variables['y'])
		self.y = np.asarray(variables['x'])

		self.a = []
		self.contact_area = []
		self.normal_force = []
		self.indentations = []
		self.fi = []
		self.Ai = []
		self.N = len(self.h)
		self.E = variables['E']
		self.nu = variables['nu']
		self.E_star = self.E/(1-self.nu**2)

		positionVectors = np.concatenate(np.asarray([self.x,self.y]).T).reshape([-1,2])
		self.rij = distance_matrix(positionVectors,positionVectors) # Precalculate distances between asperities
		self.rij[self.rij<1e-20]=float('nan')
		self.rij_2 = self.rij**2
		self.r_rep = np.transpose(np.tile(self.r,(len(self.x),1)))

	def solveNormalContact(self,indentations,convergence_criterion = 1e-12,lambda_0 = 1):
		eps = 1.0e-20

		#Initial guess for a initial indentation (overlap area)
		z = np.asarray(self.h+indentations[0])
		a = (self.r*z)**.5
		a[np.isnan(a)]=0
		u = a**2/self.r
		
		for delta in indentations:
			z = self.h+delta
			rms_error = 1
			counter = 0
			u_all = [u]
			lambda_mod = 1

			while rms_error>convergence_criterion:
				a_rep = np.transpose(np.tile(a, (len(self.x),1)))
				interaction_matrix = np.zeros(np.size(a_rep))
				interaction_matrix = (2*a_rep**2-self.rij_2)/(self.r_rep)*np.arcsin(a_rep/self.rij) + a_rep/self.r_rep*(self.rij_2 - a_rep**2)**.5 # Set up all interactions in matrix form
				interaction_matrix[np.isnan(interaction_matrix)]=0
				u = a**2/self.r + 1/np.pi*np.sum(interaction_matrix,axis=0)
				
				u_all.append(u)

				# Update contact length:
				delta_u = u-u_all[-2]
				delta_a = np.zeros(np.size(delta_u))
				delta_a[a>0] = self.r[a>0]/(2*a[a>0])*(z[a>0]-u[a>0])
				delta_a[(a+delta_a)<0] = -a[(a+delta_a)<0]

				rms_error = np.mean(delta_a**2/self.r**2)**.5
				
				# Set step length adaptively (max displacement 1% of area change):
				if sum(a>0)>0:
					if np.max(np.abs(delta_a[a>0]/self.r[a>0]))>.01:
						steplength = lambda_mod*lambda_0*.01/np.max(abs(delta_a[a>0]/self.r[a>0]))
					else:
						steplength=lambda_mod*lambda_0
				else:
					steplength=lambda_mod*lambda_0
				a = a+delta_a*steplength # Correction to asperity size
        
        		# Remove asperities no longer in contact (model predicts negative size)
				a[a<=0]=0
				#a[np.isnan(a)]=0

				# Add newly formed contacts contacts:
				a[a<eps] = (self.r[a<eps]*(z[a<eps]-u[a<eps]))**.5
				a[np.isnan(a)]=0

				#if sum((self.r[a<eps]*(z[a<eps]-u[a<eps]))*(z[a<eps]>u[a<eps])**.5)>0:
			#		print((self.r[a<eps]*(z[a<eps]-u[a<eps]))*(z[a<eps]>u[a<eps])**.5)

				counter = counter + 1

				if (counter%100)==0: # Reduce step length if convergence problems are detected
					lambda_mod = lambda_mod/2

			# append results of indentation to corresponding lists:
			self.a.append(a)
			self.Ai.append(np.pi*a**2)
			self.fi.append(4*self.E_star*a**3/(3*self.r))
			self.indentations.append(delta)
			self.contact_area.append(np.sum(self.Ai[-1]))
			self.normal_force.append(np.sum(self.fi[-1]))
			
			
        
        
   