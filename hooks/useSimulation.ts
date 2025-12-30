import { useState, useEffect, useCallback } from 'react';

export interface SimulationConfig<T> {
  initialState: T;
  tickRate?: number;
  onTick: (prevState: T, tick: number) => Partial<T>;
  onLog?: (state: T, tick: number) => string | null;
}

export function useSimulation<T extends Record<string, any>>({
  initialState,
  tickRate = 200,
  onTick,
  onLog
}: SimulationConfig<T>) {
  const [isRunning, setIsRunning] = useState(false);
  const [state, setState] = useState<T>(initialState);
  const [logs, setLogs] = useState<string[]>([]);
  const [epoch, setEpoch] = useState(0);
  
  // Keep history for numeric values automatically
  const [history, setHistory] = useState<Record<string, number[]>>({});

  useEffect(() => {
    let interval: number;
    if (isRunning) {
      interval = window.setInterval(() => {
        setEpoch(e => {
            const newEpoch = e + 1;
            
            setState(prev => {
                const updates = onTick(prev, newEpoch);
                const newState = { ...prev, ...updates };
                
                // Update History for numbers
                setHistory(prevHist => {
                   const newHist = { ...prevHist };
                   Object.keys(newState).forEach(key => {
                      if (typeof newState[key] === 'number') {
                         if (!newHist[key]) newHist[key] = [];
                         const h = newHist[key];
                         h.push(newState[key]);
                         if (h.length > 50) h.shift();
                      }
                   });
                   return newHist;
                });
      
                // Handle Logs
                if (onLog) {
                  const logMsg = onLog(newState, newEpoch);
                  if (logMsg) {
                     setLogs(prevLogs => [...prevLogs, logMsg].slice(-100)); // Keep last 100 logs
                  }
                }
                
                return newState;
            });

            return newEpoch;
        });
      }, tickRate);
    }
    return () => clearInterval(interval);
  }, [isRunning, onTick, onLog, tickRate]);

  const start = useCallback(() => setIsRunning(true), []);
  const stop = useCallback(() => setIsRunning(false), []);
  const reset = useCallback(() => {
      setIsRunning(false);
      setState(initialState);
      setEpoch(0);
      setLogs([]);
      setHistory({});
  }, [initialState]);

  return { isRunning, state, logs, history, epoch, start, stop, reset };
}