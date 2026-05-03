# Multi-stage Dockerfile for Spring Boot ML API
# Stage 1: Build stage with Maven
FROM eclipse-temurin:25-jdk-noble as builder

# Install Maven
RUN apt-get update && \
    apt-get install -y maven && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy pom.xml for dependency caching
COPY pom.xml .

# Download dependencies (cached layer)
RUN mvn dependency:go-offline -B || true

# Copy source code
COPY src ./src

# Build the application
RUN mvn clean package -DskipTests -B

# Stage 2: Runtime stage with JRE
FROM eclipse-temurin:25-jre-noble

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the JAR from builder
COPY --from=builder /build/target/*.jar app.jar

# Create directories for models and logs
RUN mkdir -p models samples logs

# Copy samples directory (models are expected to be in the classpath/resources)
# Use --chown to avoid permission issues
COPY --chown=1000:1000 samples/ ./samples/

# Expose port
EXPOSE 8080

# Health check using Spring Boot Actuator
# Note: Install curl for health checks or use wget
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/actuator/health || exit 1

# Run the application
ENTRYPOINT ["java", "-jar", "app.jar"]
