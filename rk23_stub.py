from numpy.linalg import norm
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class ODESolution:
    def __init__(self):
        self.message = "All OK"
        self.status = 0
        self.t = []
        self.y = []
        self.t_events = []
        self.y_events = []

    def _truncate_list(self, lst, max_items=3):
        """Helper function to truncate long lists with ..."""
        if len(lst) > max_items:
            return f"[{', '.join(map(str, lst[:max_items]))}, ... ({len(lst)-1} items), {lst[-1]}]"
        return str(lst)

    def __repr__(self):
        t_events_repr = self._truncate_list(self.t_events)
        y_events_repr = self._truncate_list(self.y_events)
        t_repr = self._truncate_list(self.t)
        y_repr = self._truncate_list(self.y)
        return (
            f"ODESolution(message = {self.message},\n"
            f"\tstatus = {self.status},\n"
            f"\tt = {t_repr},\n"
            f"\ty = {y_repr},\n"
            f"\tt_events = {t_events_repr},\n"
            f"\ty_events = {y_events_repr})\n"
        )

    def __str__(self):
        t_events_repr = self._truncate_list(self.t_events)
        y_events_repr = self._truncate_list(self.y_events)
        t_repr = self._truncate_list(self.t)
        y_repr = self._truncate_list(self.y)
        return (
            f"ODESolution(message = {self.message},\n"
            f"\tstatus = {self.status},\n"
            f"\tt = {t_repr},\n"
            f"\ty = {y_repr},\n"
            f"\tt_events = {t_events_repr},\n"
            f"\ty_events = {y_events_repr})\n"
        )



def rk23(f, t_range, y0, h=1, tolerance=1e-6, events=None):
    t0, t_end = t_range
    t = t0
    y = y0
    evts = []
    sol = ODESolution()

    sol.t.append(t)
    sol.y.append(y)
    iter = 0

    #print(y[0],y[1])
    while t < t_end:
        # Stage 1
        k1 = h * f(t, y)
        
        # Stage 2
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        
        # Stage 3
        k3 = h * f(t + 0.75 * h, y + 0.75 * k2)

        # Second-order estimate
        y2 = y + (2/9) * k1 + (1/3) * k2 + (4/9) * k3
        
        # Stage 4 (needed for third-order estimate)
        k4 = h * f(t + h, y2)
        
        # Third-order estimate
        y3 = y + (7/24) * k1 + (1/4) * k2 + (1/3) * k3 + (1/8) * k4

        # Error estimate (normalized)
        scaling = np.maximum(1e-10, np.linalg.norm(y))
        error = np.linalg.norm(y3 - y2) / scaling

        if error < tolerance * 1.1:
            # Accept the step
            t += h
            y = y3
            
            # Check for sign change and bracket the event
            t_prev = sol.t[-1]
            y_prev = sol.y[-1]
            if events(t, y) * events(t_prev, y_prev) < 0:
                #print(y, y_prev)
                evts.append([[t_prev, y_prev], [t, y]])
            
            sol.t.append(t)
            sol.y.append(y)
            
            # If adjustment factor is reasonable, use it; otherwise double step size.
            # Step size increase
            # Adjust step size carefully
            step_adjustment = (tolerance / error) ** (1/2)
            h = 0.9 * h + 0.1 * (h * min(1.5, step_adjustment))

            #if iter % 1000 == 0:
            #   print( t)

            iter += 1
        else:
            #print("Reducing step size...", error)
            # Reject step and reduce step size conservatively
            step_adjustment = (tolerance / error) ** (1/2)
            h = 0.9 * h + 0.1 * (h * max(0.8, step_adjustment))
        # Avoid vanishing step sizes
        h = max(h, 1e-12)

        # Prevent overshooting the endpoint
        if t + h > t_end:
            h = t_end - t

    sol.t.append(t)
    sol.y.append(y)
    
    if events:
        for evt in evts:
            te, ye = rk23_refine_event(f, events, evt, tolerance = tolerance)
            sol.t_events.append(te)
            sol.y_events.append(ye)

    return sol

def rk23_refine_event(f, events, evt, tolerance=1e-6):
    #get a better estimate for event location
    #t0, t1 = evt[0]
    #y0, y1 = evt[1]
    #evt0, evt1 = evt[2]
    # Refinement loop, probably by the secant method
    
    t0 = evt[0][0]
    y0 = evt[0][1]
    t1 = evt[1][0]
    y1 = evt[1][1]
    evt0 = events(t0,y0)
    evt1 = events(t1,y1)

    while abs(t1 - t0) > tolerance:
        t2 = t1 - evt1*(t1-t0)/(evt1-evt0)
        y2 = y0 + (t2 - t0) * (y1 - y0) / (t1 - t0)
        evt2 = events(t2,y2)
        if abs(t2-t1) < tolerance:
            #print("found zero")
            return t2,y2
        t0 = t1
        y0 = y1
        evt0 = evt1
        t1 = t2
        y1 = y2
        evt1 = evt2

    #print("found zero")
    return t1, y1

def period(list):
    #given a list of events, calclualtes the time between each event and then returns the average
    sum = 0
    for i in range(len(list)-1):
        sum += (list[i+1]-list[i])
    return sum/(len(list)-1)

if __name__ == "__main__":

    g = 9.81
    l = 3

    def f(t,y):
        return np.array([y[1], -g * np.sin(y[0]) / l])

    def event_fun(t,y):
        return y[0]

    t0, y0 = 0, np.array([2,0])
    t_end = 10
    tol = 1e-12

    sol = rk23(f, [0, t_end], y0, tolerance=tol,
               events=event_fun)
    y = sol.y[-1]
    t_events = sol.t_events
    y_events = sol.y_events
    
    sol2 = solve_ivp(
        f, 
        [t0, t_end], 
        y0, 
        method='RK23', 
        rtol=tol, 
        atol=tol, 
        events=event_fun
    )

    print("mine:", y, "solv_ivp:", sol2.y[:,-1])
    print("mine:", t_events, "solv_ivp:", sol2.t_events)
    print("mine:", y_events, "solv_ivp:", sol2.y_events)
    print("period:", period(t_events))
