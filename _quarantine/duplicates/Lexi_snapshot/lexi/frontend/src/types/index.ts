export type TaskType = 'explain' | 'fix_bugs' | 'generate_tests' | 'document' | 'optimize';

export type ModelResponse = {
  result?: {
    cleaned?: string;
    raw?: string;
    affect?: string; // <-- add this
  };
  cleaned?: string;
  raw?: string;
  affect?: string;    // <-- add this, too
};


export interface TaskResponse {
  response: string;
  task_id?: string;
  status?: 'pending' | 'completed' | 'failed';
  error?: string;
  task_type?: TaskType;
  metadata?: {
    confidence_score?: number;
    task_type?: TaskType;
    timestamp?: string;
  };
}

export interface ModelCapability {
  name: string;
  description: string;
  supported_tasks: TaskType[];
  max_context_length: number;
}

export interface TaskHistory {
  task_id: string;
  input: string;
  output: string;
  task_type: TaskType;
  timestamp: string;
  status: 'completed' | 'failed';
  metadata?: Record<string, any>;
} 
